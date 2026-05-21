use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug)]
struct Node {
    data: f64,
    children: Vec<Value>,
    local_gradients: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Value(Arc<Node>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
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
    pub fn new(data: f64) -> Self {
        Self(Arc::new(Node {
            data,
            children: Vec::new(),
            local_gradients: Vec::new(),
        }))
    }

    fn with_children(data: f64, children: Vec<Value>, local_gradients: Vec<f64>) -> Self {
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

    pub fn powf(&self, exponent: f64) -> Value {
        Value::with_children(
            self.data().powf(exponent),
            vec![self.clone()],
            vec![exponent * self.data().powf(exponent - 1.0)],
        )
    }

    pub fn log(&self) -> Value {
        Value::with_children(
            self.data().ln(),
            vec![self.clone()],
            vec![1.0 / self.data()],
        )
    }

    pub fn exp(&self) -> Value {
        let exponential = self.data().exp();
        Value::with_children(exponential, vec![self.clone()], vec![exponential])
    }

    pub fn relu(&self) -> Value {
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

    fn topological_order(&self) -> Vec<Value> {
        fn visit(node: &Value, visited: &mut HashSet<usize>, order: &mut Vec<Value>) {
            if !visited.insert(node.id()) {
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

    pub fn backward(&self) -> HashMap<usize, f64> {
        let order = self.topological_order();
        let mut gradients = HashMap::from([(self.id(), 1.0)]);

        for node in order.iter().rev() {
            let node_gradient = *gradients.get(&node.id()).unwrap_or(&0.0);
            for (child, local_gradient) in node.0.children.iter().zip(node.0.local_gradients.iter())
            {
                *gradients.entry(child.id()).or_insert(0.0) += local_gradient * node_gradient;
            }
        }

        gradients
    }

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