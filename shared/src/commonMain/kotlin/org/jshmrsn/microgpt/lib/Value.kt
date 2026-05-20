package org.jshmrsn.microgpt.lib

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow

/**
 * A scalar node in a computation graph.
 *
 * This is the entire autograd engine in miniature.
 * The blog emphasizes that autograd only needs each operation to behave like
 * a little "lego block":
 * 1) compute its forward value
 * 2) remember its inputs
 * 3) know its local derivatives with respect to those inputs
 *    (local derivatives = the partial derivative of this node's output with respect to each of its inputs,
 *     computed using only information available at this node; e.g., for z = x * y, the local derivatives
 *     are dz/dx = y and dz/dy = x, which depend only on the operation and its immediate inputs)
 *
 * Once every node does that, backpropagation is just the chain rule applied
 * systematically across the graph:
 * if parent p depends on child c, then
 * dL/dc += dL/dp * dp/dc
 * where dL/dp is the gradient arriving from above and dp/dc is this node's
 * local derivative for that edge.
 *
 * Important perspective:
 * - This implementation operates on scalars, one number at a time.
 * - PyTorch does the same algorithm over tensors (arrays of numbers), which is
 *   much faster but conceptually the same.
 */

class Value(
    var data: Double,
    children: List<Value> = emptyList(),
    localGradients: List<Double> = emptyList()
) {
    // Forward-pass numeric value at this node.

    // Gradient: derivative of the loss with respect to this node's value.
    // After backpropagation, this tells us how much the loss would change
    // if we nudged this value slightly: dL/d(this value).
    // This is the key information needed for learning: it shows which direction
    // to adjust parameters to reduce the loss.
    var gradient: Double = 0.0

    // Parents/inputs of this node in the computation graph.
    private val children: List<Value> = children

    // Local derivatives of this node with respect to each child.
    // These are edge-level terms in the chain rule:
    // dL/d(child) += dL/d(this) * d(this)/d(child)
    // Example: if z = x * y, then dz/dx = y and dz/dy = x.
    private val localGradients: List<Double> = localGradients

    /**
     * Addition node.
     *
     * If z = x + y, then:
     * dz/dx = 1
     * dz/dy = 1
     */
    operator fun plus(other: Value): Value =
        Value(data = this.data + other.data, children = listOf(this, other), localGradients = listOf(1.0, 1.0))

    operator fun plus(other: Double): Value = this + Value(other)

    /**
     * Multiplication node.
     *
     * If z = x * y, then:
     * dz/dx = y
     * dz/dy = x
     *
     * IMPORTANT NOTE ON OPERATION TRACKING:
     * These operator functions do NOT explicitly store which operation was performed
     * (e.g., no "operationType" field storing "plus" vs "times").
     *
     * Instead, the operation is implicitly encoded in the localGradients values:
     * - For addition (z = x + y): localGradients = [1.0, 1.0]
     * - For multiplication (z = x * y): localGradients = [y.data, x.data]
     * - For power (z = x^a): localGradients = [a * x^(a-1)]
     *
     * During backpropagation, we never need to "remember" what operation created
     * a node. We only need the local derivatives (localGradients) and the children.
     * The chain rule backward() method uses:
     *   child.gradient += parent.gradient * localGradient
     *
     * So the "memory" of the operation lives in the numerical values of localGradients,
     * not in any symbolic operation identifier. This is a key insight of automatic
     * differentiation: we only track local derivative values, not operation names.
     */

    operator fun times(other: Value): Value =
        Value(this.data * other.data, listOf(this, other), listOf(other.data, this.data))

    operator fun times(other: Double): Value = this * Value(other)

    /**
     * Power node.
     *
     * If z = x^a, then:
     * dz/dx = a * x^(a-1)
     */
    fun pow(other: Double): Value =
        Value(
            data.pow(other),
            listOf(this),
            listOf(other * data.pow(other - 1.0))
        )

    /**
     * Natural log.
     *
     * If z = ln(x), then:
     * dz/dx = 1/x
     */
    fun log(): Value = Value(ln(data), listOf(this), listOf(1.0 / data))

    /**
     * Exponential function: e^x
     *
     * EULER'S NUMBER (e ≈ 2.71828...):
     * e is a fundamental mathematical constant, the base of the natural logarithm.
     * It's defined as the limit: e = lim(n→∞) (1 + 1/n)^n
     *
     * Why e is special:
     * - It's the unique number where the exponential function equals its own derivative
     * - It appears naturally in compound growth, probability, and calculus
     * - exp(x) = e^x represents continuous exponential growth/decay
     *
     * EXPONENTIAL FUNCTION (e^x):
     * - Maps any real number x to a positive real number
     * - e^0 = 1 (starting point)
     * - e^1 = e ≈ 2.71828
     * - e^x grows very rapidly as x increases
     * - e^(-x) decays toward 0 as x increases
     *
     * THE UNIQUE DERIVATIVE PROPERTY:
     * The exponential function is special because it is its own derivative:
     * If z = e^x, then dz/dx = e^x
     *
     * This means:
     * - The rate of change of e^x at any point equals its value at that point
     * - This self-derivative property makes e^x fundamental in differential equations
     * - It's why e^x appears in solutions to natural growth/decay processes
     *
     * In neural networks, exp() is used in:
     * - Softmax (converting logits to probabilities)
     * - Sigmoid activations
     * - Log-likelihood computations (paired with log())
     */
    fun exp(): Value {
        // Compute e^(this.data) using Kotlin's standard library
        // e.g., if data = 2.0, then e ≈ 2.71828^2.0 ≈ 7.389
        val exponentialValue = kotlin.math.exp(data)

        // The derivative of e^x is e^x itself - this is the exponential's unique property
        // So the local gradient for backpropagation is simply the forward value we just computed
        return Value(exponentialValue, listOf(this), listOf(exponentialValue))
    }

    /**
     * ReLU activation.
     *
     * ReLU(x) = max(0, x)
     *
     * The blog notes that this micro model uses ReLU for simplicity, whereas
     * production GPT-family models often use smoother or gated activations.
     * ReLU is simpler but still captures the main idea of adding nonlinearity.
     */
    fun relu(): Value =
        Value(max(0.0, data), listOf(this), listOf(if (data > 0.0) 1.0 else 0.0))

    operator fun unaryMinus(): Value = this * -1.0
    operator fun minus(other: Value): Value = this + (-other)
    operator fun minus(other: Double): Value = this + (-other)
    operator fun div(other: Value): Value = this * other.pow(-1.0)
    operator fun div(other: Double): Value = this * Value(other).pow(-1.0)

    /**
     * Reverse-mode automatic differentiation (backpropagation).
     *
     * Why reverse mode?
     * Neural nets typically have many inputs/parameters and one scalar output loss.
     *
     *     Is 'one scalar output loss' true for LLMs?
     *     Yes, "one scalar output loss" is true for LLMs, even though they predict
     *      many tokens:
     *      - At each position, the model outputs a distribution over the vocabulary
     *        (e.g., 50,000 logits for 50,000 possible tokens)
     *      - We compute a scalar loss for that position: -log p(correct_token)
     *      - We do this for every position in the sequence
     *      - Then we AVERAGE (or SUM) these per-position losses into ONE final scalar
     *      - That single scalar is what we backpropagate through
     *
     *      So while the model has many outputs (logits), and we compute many
     *      per-position losses, these are always aggregated into a single scalar
     *      before backpropagation. This is exactly what happens in the training loop
     *      below: we collect losses[] for each position, then average them into
     *      one scalar "loss" variable, and call loss.backward() on that single scalar.
     *
     * Reverse-mode AD is efficient in exactly this regime: one backward pass gives
     * d(loss)/d(parameter) for every parameter.
     *
     * Mechanically:
     * 1) Build a topological ordering of the graph
     * 2) Seed loss.gradient = 1 because dL/dL = 1
     * 3) Traverse nodes in reverse topological order
     * 4) Propagate gradients to children using the chain rule
     *    child.gradient += parent.gradient * d(parent)/d(child)
     *
     * The blog stresses a subtle but important point:
     * gradients are accumulated with +=, not assigned.
     * If a value influences the loss through multiple paths, the total derivative
     * is the sum of the contributions from all those paths.
     */
    fun backward() {
        val topologicalOrder: List<Value> = run {
            val topologicalOrder = mutableListOf<Value>()
            val visited = mutableSetOf<Value>()

            fun buildTopologicalOrder(node: Value) {
                if (visited.add(node)) {
                    for (child in node.children) {
                        buildTopologicalOrder(child)
                    }
                    topologicalOrder.add(node)
                }
            }

            buildTopologicalOrder(this)
            topologicalOrder
        }

        // dL/dL = 1
        // d(loss)/d(loss) = 1
        this.gradient = 1.0

        // Reverse-mode automatic differentiation:
        // walk backward through the graph and distribute gradients
        // edge by edge with:
        // dL/d(child) += dL/d(parent) * d(parent)/d(child)
        // This is the multivariable chain rule, and += accumulates
        // contributions from multiple downstream paths.
        for (node in topologicalOrder.asReversed()) {
            for (childIndex in node.children.indices) {
                node.children[childIndex].gradient += node.localGradients[childIndex] * node.gradient
            }
        }
    }
}

operator fun Double.plus(other: Value): Value = Value(this) + other
operator fun Double.times(other: Value): Value = Value(this) * other
operator fun Double.minus(other: Value): Value = Value(this) + (-other)
operator fun Double.div(other: Value): Value = Value(this) * other.pow(-1.0)
