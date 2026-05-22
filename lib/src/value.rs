use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// A `Value` is one number plus the information needed to differentiate it.
// During the forward pass we build a graph of Nodes. For example, `a.mul(&b)`
// creates a new node whose data is `a * b`, whose children are `a` and `b`, and
// whose local gradients are `b` and `a`. Those local gradients are just the
// high-school-calculus slopes of the operation with respect to each input:
// d(a*b)/da = b and d(a*b)/db = a.
#[derive(Debug)]
struct Node {
    // The actual numeric value computed during the forward pass.
    data: f64,
    // Inputs that produced this value. Traversing these backward gives the
    // computation graph.
    children: Vec<Value>,
    // One slope per child. Multiplying these by the downstream gradient is the
    // chain rule.
    local_gradients: Vec<f64>,
}

// Values are reference-counted so many later computations can point back to the
// same earlier number without copying the whole graph. Equality and hashing use
// pointer identity because two nodes with the same numeric data can represent
// different parameters or different intermediate computations.
#[derive(Clone, Debug)]
pub struct Value(Arc<Node>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Pointer identity is the right equality here. Two independent leaves can
        // both contain 0.0, but their gradients must remain separate if one is a
        // query bias and the other is a feed-forward bias. Treating equal numeric
        // data as the same node would merge unrelated parameters during backward.
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl Value {
    // A leaf value has no children. Model parameters and constants start here.
    pub fn new(data: f64) -> Self {
        // Leaf nodes are the "inputs" of the computation graph. A model weight,
        // a bias, or a literal constant such as 1.0 all begin as a node with no
        // parents. Later operations build new nodes that point back to these
        // leaves, allowing the final loss to trace how each leaf contributed.
        Self(Arc::new(Node {
            data,
            children: Vec::new(),
            local_gradients: Vec::new(),
        }))
    }

    // Internal constructor for operations. `local_gradients[i]` must correspond
    // to `children[i]`.
    fn with_children(data: f64, children: Vec<Value>, local_gradients: Vec<f64>) -> Self {
        // Each operation stores only local information: its output value, the
        // input nodes, and the derivative of this output with respect to each
        // input. The global derivative of the final loss is not known yet. The
        // backward pass combines these local slopes later using the chain rule.
        Self(Arc::new(Node {
            data,
            children,
            local_gradients,
        }))
    }

    pub fn data(&self) -> f64 {
        self.0.data
    }

    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    // Addition sends the same downstream gradient to both inputs because
    // d(a+b)/da = 1 and d(a+b)/db = 1.
    pub fn add(&self, other: &Value) -> Value {
        Value::with_children(
            self.data() + other.data(),
            vec![self.clone(), other.clone()],
            vec![1.0, 1.0],
        )
    }

    pub fn add_f64(&self, other: f64) -> Value {
        self.add(&Value::new(other))
    }

    // Multiplication is where the chain rule becomes visible: the slope with
    // respect to the left input is the right input, and vice versa.
    pub fn mul(&self, other: &Value) -> Value {
        Value::with_children(
            self.data() * other.data(),
            vec![self.clone(), other.clone()],
            vec![other.data(), self.data()],
        )
    }

    pub fn mul_f64(&self, other: f64) -> Value {
        self.mul(&Value::new(other))
    }

    // Power, log, and exp provide enough calculus to build softmax,
    // cross-entropy, normalization, and optimizer math.
    pub fn powf(&self, exponent: f64) -> Value {
        // d(x^n)/dx = n*x^(n-1). In this project powers appear in RMSNorm
        // (`x^2` and inverse square root), Adam (`sqrt` can be expressed as a
        // power in the scalar backend), and division (`x^-1` in `div` below).
        Value::with_children(
            self.data().powf(exponent),
            vec![self.clone()],
            vec![exponent * self.data().powf(exponent - 1.0)],
        )
    }

    pub fn log(&self) -> Value {
        // Log shows up in cross-entropy. Its derivative, 1/x, is large when x is
        // small, which matches the learning signal we want: assigning tiny
        // probability to the correct token should produce a strong correction.
        Value::with_children(
            self.data().ln(),
            vec![self.clone()],
            vec![1.0 / self.data()],
        )
    }

    pub fn exp(&self) -> Value {
        let exponential = self.data().exp();
        // exp is its own derivative. Softmax uses exp to turn relative logits
        // into positive unnormalized probabilities before dividing by their sum.
        Value::with_children(exponential, vec![self.clone()], vec![exponential])
    }

    pub fn relu(&self) -> Value {
        // ReLU is piecewise linear: it passes positive values and zeros negative
        // values. The local derivative is therefore 1 on the positive side and 0
        // on the negative side, so negative activations stop sending gradient
        // backward through this path.
        Value::with_children(
            self.data().max(0.0),
            vec![self.clone()],
            vec![if self.data() > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    pub fn neg(&self) -> Value {
        self.mul_f64(-1.0)
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.powf(-1.0))
    }

    pub fn div_f64(&self, other: f64) -> Value {
        self.div(&Value::new(other))
    }

    // Reverse-mode autodiff needs children to be visited before parents when
    // preparing the backward pass. This topological order is like listing all
    // recipe ingredients before the final dish.
    fn topological_order(&self) -> Vec<Value> {
        fn visit(node: &Value, visited: &mut HashSet<usize>, order: &mut Vec<Value>) {
            if !visited.insert(node.id()) {
                // A node can feed into the loss through multiple later paths.
                // Visiting it once prevents exponential work and keeps the
                // topological list unique; the backward pass still accumulates
                // all path contributions with `+=`.
                return;
            }
            for child in &node.0.children {
                visit(child, visited, order);
            }
            order.push(node.clone());
        }

        let mut visited = HashSet::new();
        let mut order = Vec::new();
        visit(self, &mut visited, &mut order);
        order
    }

    // Compute d(self)/d(each node). Start with d(self)/d(self) = 1, then walk
    // backward through the graph. Each edge applies:
    //
    // child_gradient += parent_gradient * local_gradient
    //
    // That is the chain rule in code. If a node influences the loss through
    // multiple paths, the `+=` adds those paths together.
    pub fn backward(&self) -> HashMap<usize, f64> {
        let order = self.topological_order();
        let mut gradients = HashMap::from([(self.id(), 1.0)]);

        for node in order.iter().rev() {
            // `node_gradient` is the derivative of the final output (`self`)
            // with respect to this node. For each child edge, multiply by the
            // local derivative stored during the forward operation:
            //
            // d(loss)/d(child) += d(loss)/d(node) * d(node)/d(child)
            //
            // The map is sparse because many constants or intermediates may not
            // influence the final scalar in a particular path.
            let node_gradient = *gradients.get(&node.id()).unwrap_or(&0.0);
            for (child, local_gradient) in node.0.children.iter().zip(node.0.local_gradients.iter())
            {
                *gradients.entry(child.id()).or_insert(0.0) += local_gradient * node_gradient;
            }
        }

        gradients
    }

    // Training only needs gradients for model parameters, not every temporary
    // value in the graph. The caller gives us the parameter node ids and we
    // return a dense vector aligned with the model's flattened parameter order.
    pub fn backward_for(
        &self,
        parameter_index_by_value: &HashMap<usize, usize>,
        parameter_count: usize,
    ) -> Vec<f64> {
        let gradients_by_id = self.backward();
        let mut gradients = vec![0.0; parameter_count];

        for (value_id, parameter_index) in parameter_index_by_value {
            gradients[*parameter_index] = *gradients_by_id.get(value_id).unwrap_or(&0.0);
        }

        gradients
    }
}
