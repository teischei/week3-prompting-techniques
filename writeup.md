# Week 3 Prompting Techniques Writeup

All four checks were run locally with `llama3.1:8b`.

## 1. K-shot Prompting: Ticket Classification

### Final Prompt

```text
You classify online-store support tickets. Return exactly one lowercase label
and nothing else: billing, shipping, returns, technical, or other.

Examples:
Ticket: I was billed twice after checking out.
Label: billing

Ticket: The tracking page has not updated and my delivery is late.
Label: shipping

Ticket: These headphones do not fit; how do I send them back?
Label: returns

Ticket: I get an error whenever I try to sign in.
Label: technical

Ticket: What colors does this product come in?
Label: other

Match the new ticket to the closest example and reply with only its label.
```

### Sample Input and Output

Input: `My card was charged twice for order 4412.`  
Output: `billing`

The built-in check passed all four tickets (`4/4 correct`).

**What I learned:** Examples show the model both how to distinguish similar support requests and
the exact short format expected by the program.

## 2. Chain-of-Thought: Refund Eligibility

### Final Prompt

```text
You are a careful support-desk assistant deciding whether an order qualifies
for a refund. Work through the policy criteria in order: purchase timing, item
condition, and whether it is final sale. A refund is allowed only when every
requirement is satisfied. Explain your brief reasoning, then put the result on
the final line in exactly one of these forms:
DECISION: YES
DECISION: NO
Never put any text after that final decision line.
```

### Sample Input and Output

Input: `Bought 10 days ago, unused, not final sale.`  
Output: a short policy review ending in `DECISION: YES` (parsed by the check as `YES`)

The built-in check passed all three policy scenarios (`3/3 correct`).

**What I learned:** Asking the model to consider each policy condition prevents it from deciding
from only one detail, while the fixed last line makes the answer easy to check automatically.

## 3. Tool Calling: Order Status Lookup

### Final Prompt

```text
You are an online-store support assistant. When a customer asks for an
order's delivery or processing status and provides an order ID, call
get_order_status with that exact ID before responding. Use the tool result for
your answer and never invent an order status.
```

### Tool Schema Details

- Tool: `get_order_status`
- Description: retrieves the current fulfillment status for a store order by order ID.
- Required parameter: `order_id`, the ID supplied by the customer.

### Sample Input and Output

Input: `Hi, can you check on order 4412 for me?`  
Output: a call to `get_order_status(order_id="4412")`, the local tool returned
`shipped, arriving Friday`.

The built-in check passed and verified the exact order ID.

**What I learned:** The model needs a clear function description and argument definition before it
can reliably choose a tool instead of trying to guess an answer from the conversation alone.

## 4. Retrieval-Augmented Generation: Help-Center Answers

### Final Prompt

```text
You are a support-desk assistant. Answer only with facts supported by the
help-center context supplied in the user's message. Do not rely on outside
knowledge or make up policy details. If the context is empty or does not
answer the question, respond politely: "I don't know based on the help-center
information provided." Keep supported answers concise.
```

### Retriever

The retriever normalizes related words such as `returns` and `return`, as well as `shipping`
and `ship`. It scores each help-center line by its overlapping terms with the question and passes
only matching lines to the model.

### Sample Input and Output

Input: `How long do I have to return an item?`  
Output: `you have 30 days from the delivery date to return an item for a full refund.`

For the unsupported question `What's your CEO's home address?`, the model replied:
`i don't know based on the help-center information provided.`

The built-in check passed all three cases (`3/3 correct`).

**What I learned:** Retrieval narrows the information available to the model, and the explicit
fallback instruction helps it decline questions that the supplied documentation cannot answer.

## Overall Reflection

LLMs are effective at structured tasks when the task boundaries and output shape are made explicit.
Examples help with classification, a checklist helps apply a policy consistently, and a tool schema
gives the model a concrete action to take. They still need scaffolding when correctness depends on
specific facts. The RAG context limits the available evidence, and the refusal rule avoids confident
answers when evidence is missing. Across the four techniques, I saw that prompt structure is what
turns a general text generator into a more predictable support assistant.

## Commands and Results

Run from this `starter/` directory:

```sh
python3 k_shot_prompting.py
python3 chain_of_thought.py
python3 tool_calling.py
python3 rag.py
```

Final results: K-shot `4/4 correct`, chain-of-thought `3/3 correct`, tool calling `PASS` with
`order_id='4412'`, and RAG `3/3 correct`.
