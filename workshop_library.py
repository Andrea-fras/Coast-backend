"""Coast's curated workshops: hand-written milestone contracts, the labs each milestone
uses, and a short reference of checked facts so Pedro teaches from numbers we verified
rather than from memory.

A milestone contract: title, outcome (what the student makes), criteria (evidence in the
student's own work), coaching (how Pedro runs it), minutes, topics (concept names for
grading tags), tools (labs Pedro can place: id, params, when to use it, and for a Python lab
what its starter code contains) and reference.
The labs live in the frontend (src/widgets); their numbers here match the simulators, and each
Python lab's "code" note matches its starter in src/widgets/python/labs.js.
"""

LLM = {
    "outcome": "A tiny language model you built and trained yourself in Python, and a clear picture of how ChatGPT turns text into tokens, predicts the next one, and chooses what to write.",
    "steps": [
        {
            "title": "Turn text into tokens",
            "outcome": "A tokenizer for text you chose: encode turns it into numbers, decode turns them back.",
            "criteria": [
                "Explain why a language model needs text as numbers, and what one token is in a character tokenizer compared with a GPT-style subword tokenizer.",
                "Write encode and decode so the Python lab's round-trip tests pass on their own text.",
            ],
            "topics": ["tokens", "tokenizer", "byte-pair encoding"],
            "tools": [
                {"id": "tokens", "params": {}, "use": "First, before any code: they paste their own text and compare characters, words and subword pieces."},
                {"id": "python", "params": {"lab": "tokenizer"}, "use": "Once they can say what a token is: they fill in stoi, itos, encode and decode.",
                 "code": "The starter makes chars (the text's distinct characters, sorted) and two empty dictionaries, stoi (character to number) and itos (number to character). The loops are written; each TODO is one line inside one. TODO 1, inside for i, c in enumerate(chars): store i under c in stoi and c under i in itos. TODO 2, in encode: append each character's number. TODO 3, in decode: append each number's character; \"\".join then glues them into a string. Python it needs: a dictionary (d[key] = value stores, d[key] reads), enumerate (gives 0 and the first item, 1 and the second, ...), list.append."},
            ],
            "coaching": "Open with the surprise that a model never sees letters, only numbers. Let them explore the tokenizer lab first and ask what they notice (why do rare words break into pieces? which view gives the fewest tokens, and what does that cost?). Then the Python lab: walk through what the starter does, and explain a dictionary (a phone book: look up a name, get a number) and enumerate with tiny unrelated examples before they write anything; never write their stoi, encode or decode lines. The lab's tests plus their explanation are the evidence.",
            "reference": "A token is the unit a model reads and writes. GPT tokenizers use byte-pair encoding (BPE): start from bytes or characters and repeatedly merge the most frequent neighbouring pair into a new token. GPT-2's vocabulary has 50,257 tokens; later OpenAI tokenizers have about 100,000 (cl100k) and about 200,000 (o200k). In English a token averages about 4 characters, roughly three quarters of a word. A character tokenizer has a tiny vocabulary (a few dozen symbols for English) but long sequences; a word tokenizer has short sequences but a huge vocabulary and no way to spell new words. The lab's subword view learns merges from the student's text only, so its pieces differ from GPT's.",
            "minutes": 20,
        },
        {
            "title": "Predict the next character",
            "outcome": "A bigram model: a table of which character follows which, and new text generated from it.",
            "criteria": [
                "Count which character follows which, turn a row into probabilities, and pass the Python lab's tests.",
                "Generate text and explain why it looks word-like but means nothing, in terms of how much context the model sees.",
            ],
            "topics": ["bigram model", "next-token prediction", "probability"],
            "tools": [
                {"id": "python", "params": {"lab": "bigram"}, "use": "After a one-paragraph explanation of predicting the next character from counts.",
                 "code": "The starter has the tokenizer from the first lab written compactly (comprehensions, explained in its comments), counts as a V-by-V table of zeros (a list of lists: counts[row][column]) and generate, which samples with random.choices. TODO 1, inside for a, b in zip(text, text[1:]) (zip pairs each character with the next one): add one to the cell whose row is a's number and column is b's. TODO 2, inside a loop over one row: append each count divided by the row total. Until TODO 1 is done it prints a reminder instead of text."},
            ],
            "coaching": "Connect it to phone autocomplete: a language model is a next-token predictor. Before they code, ask which character they expect most often after 't' in the sonnets, then check it against their counts. Show the table on a tiny case first: in 'abab' the pairs are ab, ba, ab, so the cell for row a, column b holds 2. Hint the counting line with 'which row, which column?' rather than giving it. After they generate text, the key insight: the model only ever sees one previous character. Ask what would make it better (more context), which sets up attention later.",
            "reference": "A bigram model estimates P(next | current) as count[current][next] divided by the row total, and generates by sampling from those probabilities again and again. It learns letter patterns (frequent pairs like 'th', a capital after a newline) but not words or meaning, because its context is a single character. GPT-style models condition on a long window of previous tokens: thousands to hundreds of thousands in current systems.",
            "minutes": 25,
        },
        {
            "title": "Temperature: why answers vary",
            "outcome": "Sampling with temperature in their generator, and a tested prediction of what low and high temperature do.",
            "criteria": [
                "Predict what very low and very high temperature do to the choice of next word, then test it in the temperature lab and explain what they saw.",
                "Add temperature to their sampling code and show output at a low and a high setting, with the Python lab's tests passing.",
            ],
            "topics": ["temperature", "sampling", "softmax"],
            "tools": [
                {"id": "temperature", "params": {}, "use": "Right after asking for their prediction, and not before: they slide, draw ten next words at different temperatures, and send their draws."},
                {"id": "python", "params": {"lab": "temperature"}, "use": "Once they can explain the effect: they implement it in their own generator.",
                 "code": "The starter has the tokenizer, the counts and next_char_probs from the bigram lab, and generate printing text at T = 0.3 and T = 2.0. with_temperature has two loops that copy the numbers unchanged. TODO 1: append p raised to the power 1 / T (in Python p ** (1 / T)). TODO 2: append each value divided by their total. Python it needs: ** for a power."},
            ],
            "coaching": "Ask for the prediction before they touch the slider. Use both scenarios, a fact with one right answer and a story with many good continuations, and ask which temperature they would choose for each and why. In code: raise each probability to the power 1/T, then divide by the new total so they sum to 1 again. Work one tiny case with them first: at T = 0.5, [0.8, 0.2] becomes 0.64 and 0.04, which divided by 0.68 is about 0.94 and 0.06, so the likelier choice gets likelier. Connect it to ChatGPT: the same prompt can give different answers because every word is sampled.",
            "reference": "Temperature T divides the model's scores (logits) before the softmax: p_i is proportional to exp(z_i / T), which is the same as raising the probabilities to the power 1/T and renormalising. As T approaches 0 the most likely token is always chosen (greedy decoding: repeatable, prone to loops); T = 1 samples the model's own distribution; T above 1 flattens it (more varied, more mistakes). Chat systems usually also limit sampling to the most likely tokens (top-k or top-p, 'nucleus', sampling). The lab's scores are illustrative, not taken from a real model.",
            "minutes": 15,
        },
        {
            "title": "Attention: which words matter",
            "outcome": "One attention head computed in code, and an explanation of what a word attends to and why.",
            "criteria": [
                "Use the attention lab to find what 'tired' and 'wide' attend to, explain it with queries and keys, and explain why 'it' can't decide yet.",
                "Compute causal scaled dot-product attention in the Python lab so its tests pass, and say what the causal mask prevents.",
            ],
            "topics": ["attention", "queries and keys", "causal mask"],
            "tools": [
                {"id": "attention", "params": {}, "use": "First: they click words, switch between the 'tired' and 'wide' endings, and send what they found."},
                {"id": "python", "params": {"lab": "attention"}, "use": "Then the mechanics in numpy.",
                 "code": "The starter gives six words as made-up vectors of four numbers (the rows of X), a head's query and key matrices, and Q and K. In attention_weights the causal mask is already applied: it sets each later word's score to minus infinity. TODO 1: scores = Q @ K.T / np.sqrt(d). TODO 2: softmax each row: np.exp of the scores, divided by each row's total. Python it needs: a numpy array as a table of numbers; @ multiplies matrices (here: every query dotted with every key); .T turns K on its side; np.exp; .sum(axis=1, keepdims=True) totals each row (axis=1 means along a row; keepdims keeps the totals lined up with the rows for the division). e to the minus infinity is 0, so later words get no weight."},
            ],
            "coaching": "Start from the bigram's weakness: it saw one character. Attention lets every position look back at all earlier ones and choose which matter. In a GPT a word only sees earlier words, so 'it' can't know yet whether it means the animal or the street; the last word looks back and finds the right noun. Keep the maths to one line: score = query · key / sqrt(d), softmax, the weights add up to 1. The Python lab's numbers are made up: the point is the mechanics. Numpy is new to most students: before the lab, show a 2-by-2 example of @ (every row of one table dotted with every row of the other) and of totalling each row.",
            "reference": "Scaled dot-product attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V (Vaswani et al., 2017, 'Attention Is All You Need'). Each token has a query (what it looks for), a key (what it offers) and a value (what it passes on). In decoder-only models like GPT a causal mask stops each position attending to later ones, which is what lets them be trained to predict the next token. Dividing by sqrt(d_k) keeps scores from growing with vector size, which would make the softmax too peaked. Models run many heads per layer and stack many layers. The lab's word vectors are hand-made to show the idea; real ones are learned. In the lab, 'tired' gives about 85% of its attention to 'animal' and 'wide' about 85% to 'street', while 'it' splits evenly between the two (about 36% each).",
            "minutes": 20,
        },
        {
            "title": "Train your model and make it write",
            "outcome": "A small neural model trained on text they chose, with its loss falling, and text it generated.",
            "criteria": [
                "Write the loss and the update step so training runs and the Python lab's tests pass, and explain what the loss measures and why it starts near log(V).",
                "Generate text from the trained model and explain two ways a model like ChatGPT differs from theirs.",
            ],
            "topics": ["training", "loss", "gradient descent"],
            "tools": [
                {"id": "python", "params": {"lab": "train"}, "use": "After explaining the training loop in words: predict, measure the error, nudge the weights, repeat.",
                 "code": "The starter gives xs and ys (each character's number and the next character's), W (one row of V scores per character, starting near zero), softmax, and a loop of 300 steps that already computes the probabilities and the gradient. The gradient lines are given and need no edits: describe them as how much each score should move to lower the loss. TODO 1, the loss: probs[np.arange(N), ys] picks, for every position, the probability given to the character that really came next (row i, column ys[i]); take -np.log of those and their .mean(). TODO 2: W -= learning_rate * dW. Python it needs: numpy indexing with two arrays, np.log, .mean()."},
            ],
            "coaching": "This is the payoff: the same loop trains GPT. Walk through it in words first, then show on a small table how probs[np.arange(N), ys] picks one number from each row before they write the loss. The starting loss near ln(V) is a check they can reason about: guessing at random among V characters. Invite them to paste a longer text of their own (a few thousand characters) and compare what it writes. For the last criterion look for real differences: a transformer with attention instead of one table of scores, vastly more parameters and data, long context, and fine-tuning on instructions and human feedback.",
            "reference": "Cross-entropy loss is the average of -log(probability the model gave to the character that actually came next). Guessing uniformly among V characters gives a loss of ln(V): the lab's default text (two Shakespeare sonnets) has 44 different characters, so about 3.78. Gradient descent moves every weight a small step against the gradient of the loss. The lab's model is a neural bigram, one row of scores per character; after training it learns roughly the probabilities the counting model had, which is why its text looks similar. GPT-3 had 175 billion parameters and was trained on about 300 billion tokens (Brown et al., 2020). Assistants like ChatGPT are then fine-tuned on instructions and with human feedback (RLHF) to follow requests. The details of the newest models are not public.",
            "minutes": 30,
        },
    ],
}

ROCKET = {
    "outcome": "A rocket you designed and flew in a physics simulator that carries a 10 kg payload past the 100 km edge of space, and the physics to explain every choice.",
    "steps": [
        {
            "title": "Why rockets fly",
            "outcome": "A rocket that lifts off, and the reason it does.",
            "criteria": [
                "Explain what a rocket pushes against to move, and why it works in the vacuum of space.",
                "Predict, then show in the rocket lab, one design that stays on the pad and one that lifts off, and explain the difference with the thrust-to-weight ratio.",
            ],
            "topics": ["Newton's third law", "thrust", "thrust-to-weight ratio"],
            "tools": [
                {"id": "rocket", "params": {"scene": "liftoff"}, "use": "Once they have given their own idea of how a rocket moves. Its starting design stays on the pad."},
            ],
            "coaching": "Many people think rockets push against the air or the ground: ask what they think first. Then Newton's third law: the engine throws exhaust down, the exhaust pushes the rocket up. Before the lab, say that thrust and weight are both forces measured in kilonewtons (kN), and that 1 kN holds up about 102 kg, so they can compare the two numbers. The lab's starting design (1,000 kg, 9 kN of thrust) stays on the pad; they predict before every launch, then change thrust or mass until it lifts off, and state the rule in their own words.",
            "reference": "Newton's third law: the engine pushes exhaust backwards and the exhaust pushes the rocket forwards, so no air is needed (air only slows a rocket down). Thrust is roughly mass flow rate times exhaust velocity (plus a pressure term the lab ignores). Weight = mass x g, with g = 9.81 m/s^2. A rocket lifts off only if thrust is greater than weight, a thrust-to-weight ratio above 1; launchers usually lift off at about 1.2 to 1.5. In the lab 1 kN of thrust holds up about 102 kg, so the 1,000 kg starting rocket (9 kN) stays on the pad and needs more than 9.81 kN.",
            "minutes": 15,
        },
        {
            "title": "The rocket equation",
            "outcome": "The delta-v of their design, worked out by them and checked in the lab.",
            "criteria": [
                "Calculate their rocket's delta-v with the rocket equation and match the lab's value within a few percent.",
                "Predict what doubling the fuel does to delta-v, test it, and explain the result using the logarithm and the heavier tanks.",
            ],
            "topics": ["delta-v", "rocket equation", "exhaust velocity", "mass ratio"],
            "tools": [
                {"id": "rocket", "params": {"scene": "design"}, "use": "After introducing delta-v and the equation: they compute first, enter their prediction, then press Check."},
            ],
            "coaching": "Introduce delta-v as the rocket's budget of speed change. Give the equation, say what each symbol means and what ln does (the calculator's ln button; it grows more and more slowly, which is why extra fuel helps less and less), work it once on different numbers (exhaust 2,000 m/s, 300 kg full, 100 kg once the fuel is gone: 2,000 x ln 3 = about 2,197 m/s), then let them compute the starting design with a calculator before pressing Check (the lab shows its full and empty masses), and ask what each part means. Then the surprise: doubling the fuel does not double delta-v, because the extra fuel must also be carried (and bigger tanks weigh more). Ask for a prediction first. The engine choice (exhaust velocity) scales delta-v directly.",
            "reference": "Tsiolkovsky rocket equation: delta-v = v_e x ln(m0 / mf), with v_e the exhaust velocity, m0 the full mass and mf the mass once the fuel is gone. Lab engines: solid motor 2,400 m/s, kerosene + oxygen 2,900 m/s, hydrogen + oxygen 4,200 m/s (specific impulse Isp = v_e / 9.81, so 2,900 m/s is about 296 s). The lab's starting design: 1,000 kg with 600 kg of fuel, m0/mf = 2.5, delta-v = 2,900 x ln 2.5 = about 2,657 m/s. Doubling the fuel to 1,200 kg with the same 400 kg of tanks and engine gives m0/mf = 4 and about 4,020 m/s: 51% more, not twice. The lab requires tanks to weigh at least a quarter of their fuel, plus 1 kg of engine per kN of thrust.",
            "minutes": 20,
        },
        {
            "title": "Gravity and air",
            "outcome": "A flight to the highest point they can reach with one stage, and a measured account of where the delta-v went.",
            "criteria": [
                "Predict the highest point of their design, launch it, and explain the gap between the ideal delta-v and the speed at burnout (gravity loss and drag loss).",
                "Change one design choice (thrust, diameter or fuel) to cut the losses, and show the improvement in the lab.",
            ],
            "topics": ["gravity loss", "drag", "apogee"],
            "tools": [
                {"id": "rocket", "params": {"scene": "flight"}, "use": "Once they know delta-v: they predict the highest point, launch, and read the losses."},
            ],
            "coaching": "Ideal delta-v assumes empty space. Ask for a rough prediction of the highest point: any guess is fine, with airliners at about 10 km and space at 100 km as reference points. A student comfortable with formulas can estimate v^2 / 2g from the ideal delta-v, which shows why it's too optimistic. Discuss the gap after the flight. The lab reports gravity and drag losses: let them find the trade-off themselves. More thrust burns faster, so less gravity loss, but reaches high speed in thick air (more drag, more g); a thinner rocket has less drag.",
            "reference": "While the engine burns, speed at burnout = ideal delta-v - gravity loss - drag loss (exactly, in the lab's vertical flight). Gravity loss is about g x burn time for a vertical climb, so slow burns waste delta-v. Drag = 1/2 x air density x v^2 x drag coefficient x area; air density falls by a factor of e about every 8.5 km (1.225 kg/m^3 at sea level), so drag matters most low down. After burnout the rocket coasts up until its speed is zero; without air that adds about v^2 / (2g) of height. Space conventionally starts at 100 km (the Karman line). The lab's starting design (800 kg, 0.5 m wide, 15 kN) peaks at about 73 km: its ideal 2,010 m/s shrinks to about 940 m/s at burnout, with about 760 m/s lost to gravity and 320 m/s to drag.",
            "minutes": 20,
        },
        {
            "title": "Staging",
            "outcome": "A two-stage rocket that flies higher than one stage of the same total mass.",
            "criteria": [
                "Explain with the rocket equation why dropping an empty stage increases delta-v.",
                "Design a two-stage rocket, compare it in the lab with one stage of the same total mass, and show it flies higher.",
            ],
            "topics": ["staging", "mass ratio", "delta-v"],
            "tools": [
                {"id": "rocket", "params": {"scene": "staging"}, "use": "After asking why real rockets drop parts: the lab automatically flies the same mass as one stage for comparison."},
            ],
            "coaching": "Ask why real rockets drop parts, and lead them to it: once a tank is empty, carrying it wastes fuel. Encourage experiments with how to split the mass between the stages, and ask them to explain what they found (the upper stage usually works best much smaller than the first).",
            "reference": "Each stage's delta-v comes from the rocket equation with everything above it counted as payload; the total delta-v is the sum. Dropping empty tanks raises the later stages' mass ratios, so the same propellant gives more delta-v. The lab's starting two-stage design (1,000 kg in all) reaches about 337 km, while the same 1,000 kg as one stage reaches about 167 km. Real orbital launchers use two or three stages: Saturn V had three, Falcon 9 has two.",
            "minutes": 20,
        },
        {
            "title": "Mission: reach space",
            "outcome": "A rocket that carries a 10 kg payload past 100 km, as light as they can make it, with a short flight report.",
            "criteria": [
                "Fly a design in the mission lab that carries the 10 kg payload above 100 km.",
                "Explain their design choices using delta-v, losses and thrust-to-weight, and name what a rocket would additionally need to reach orbit rather than just space.",
            ],
            "topics": ["mission design", "orbital velocity", "delta-v budget"],
            "tools": [
                {"id": "rocket", "params": {"scene": "mission", "payload": 10}, "use": "The design challenge: they iterate until the payload reaches space, then try to make the rocket lighter."},
            ],
            "coaching": "Their design, their reasoning: coach with questions, never with a finished design. Once they succeed, push for a lighter rocket. For the orbit question: reaching space means going up; staying in orbit means going sideways fast enough, about 7.8 km/s, to keep falling around the Earth.",
            "reference": "A sounding rocket goes up and falls back; an orbital rocket must also reach about 7.8 km/s sideways at low Earth orbit altitude, which takes roughly 9.3 to 10 km/s of delta-v including gravity and drag losses, far more than a vertical trip to 100 km. The lab flies straight up, so it can reach space but not orbit. The lab's starting design (one stage, 610 kg with the payload) peaks at about 55 km, short of space. Much lighter designs, well under 100 kg, can reach space in the lab, because mass ratio and drag matter more than size: let them discover this, don't hand them a design.",
            "minutes": 25,
        },
    ],
}

BRAIN = {
    "outcome": "A simulated neuron you drove and measured yourself, a circuit that detects coincidences, and a synapse that learns an association, with the biology behind each.",
    "steps": [
        {
            "title": "A neuron at rest",
            "outcome": "A neuron model they can drive with current, and a tested prediction of when it fires.",
            "criteria": [
                "Explain the resting potential and why the membrane drifts back to rest (the leak).",
                "Predict for a chosen current whether the neuron fires, test it in the neuron lab, and explain the result using the threshold.",
            ],
            "topics": ["resting potential", "membrane", "threshold"],
            "tools": [
                {"id": "neuron", "params": {"mode": "single"}, "use": "After the leaky-bucket picture: they predict, then inject current."},
            ],
            "coaching": "Give the picture of a leaky bucket: input current fills it, the leak drains it, and the neuron fires when the level reaches threshold. Say what the units are the first time: the membrane in millivolts (mV, thousandths of a volt), the input current in nanoamps (nA). Ask for a prediction before each run. Good contrasts: 1 nA (the membrane rises to -60 mV and no spike) against 2 nA (it fires). Ask them to find the smallest current that fires and explain why it is 1.5 nA here.",
            "reference": "Resting potential is about -70 mV (typically -60 to -80 mV), maintained by ion pumps and mostly by potassium leak channels. The lab uses a leaky integrate-and-fire model: tau x dV/dt = -(V - V_rest) + R x I, with tau = 15 ms and R = 10 megaohms, so a constant 1 nA settles the membrane 10 mV above rest (-60 mV). Threshold is -55 mV; the neuron fires once R x I exceeds 15 mV, that is above 1.5 nA. Real action potentials come from voltage-gated sodium channels opening (sodium rushes in) and potassium channels then repolarising; the model replaces all of that with a threshold and a reset.",
            "minutes": 15,
        },
        {
            "title": "All-or-nothing spikes",
            "outcome": "The neuron's firing-rate curve, predicted and then measured.",
            "criteria": [
                "Explain why a spike is all-or-nothing and how a neuron signals a stronger input.",
                "Guess the smallest current that makes it fire, measure the rate at four or more currents, and explain the shape of the curve (the threshold, and why the rate levels off).",
            ],
            "topics": ["action potential", "firing rate", "refractory period"],
            "tools": [
                {"id": "neuron", "params": {"mode": "rate"}, "use": "After asking them to describe the curve they expect."},
            ],
            "coaching": "Key idea: spikes don't grow with a stronger input, they come more often (a rate code). Ask them to describe the curve they expect before measuring; words are enough ('nothing, then rising slowly'). Hz means spikes per second. Afterwards ask why nothing happens below 1.5 nA and why the rate can't grow forever.",
            "reference": "Action potentials are all-or-none: once threshold is crossed the spike has the same size. Stronger input raises the firing rate. In the lab the rate for a constant current I above threshold is 1 / (t_ref + tau x ln(RI / (RI - 15 mV))): about 44 Hz at 2 nA, 81 Hz at 3 nA, 136 Hz at 5 nA and 225 Hz at 10 nA. The 2 ms refractory period keeps the rate below 500 Hz. Real neurons have absolute refractory periods of about 1 to 2 ms.",
            "minutes": 15,
        },
        {
            "title": "Synapses: adding up inputs",
            "outcome": "A neuron that fires only when two inputs arrive together.",
            "criteria": [
                "Explain excitatory and inhibitory synapses and how inputs add up in time.",
                "Set the strengths and timing in the lab so the neuron fires for both inputs together but not for either alone, find the largest delay that still works, and show what an inhibitory input does.",
            ],
            "topics": ["synapse", "summation", "coincidence detection", "inhibition"],
            "tools": [
                {"id": "neuron", "params": {"mode": "synapses"}, "use": "After explaining summation: they predict each trial, then run it."},
            ],
            "coaching": "Neurons add up their inputs: each input briefly raises (or, if inhibitory, lowers) the membrane. Ask for a prediction before each run. With strength 5, one input alone reaches about -60 mV and two together fire; the window closes between 15 and 20 ms apart. This is a coincidence detector, like the auditory neurons that compare when a sound reaches each ear.",
            "reference": "Excitatory synapses (mostly glutamate) depolarise the membrane (an EPSP); inhibitory ones (mostly GABA, and glycine in the spinal cord) push it away from threshold (an IPSP). A single real synapse gives an EPSP of roughly 0.1 to 1 mV, so a neuron needs many inputs together; the lab's 'strength' stands for a volley from many synapses (strength 5 gives a bump of about 9.7 mV). Inputs fade over milliseconds (5 ms for the synaptic current, 15 ms for the membrane in the lab), so only inputs close in time reach threshold together (temporal summation). Neurons in the auditory brainstem use coincidence detection to compare a sound's arrival time at the two ears.",
            "minutes": 20,
        },
        {
            "title": "Learning: fire together, wire together",
            "outcome": "A synapse that learned to link a bell to food, and the rule that made it learn.",
            "criteria": [
                "Explain Hebb's idea and the role of the reward signal in the lab's learning rule.",
                "Pair bell and food until the bell alone fires the neuron, report how many pairings it took, and explain why the learning happened.",
            ],
            "topics": ["Hebbian learning", "synaptic plasticity", "classical conditioning"],
            "tools": [
                {"id": "neuron", "params": {"mode": "learning"}, "use": "After Hebb's idea: they predict how many pairings it will take, then train and test the bell alone."},
            ],
            "coaching": "Link it to Pavlov's dogs. Before training ask whether the bell alone fires the neuron now, and after how many pairings it will. The rule: when the bell's input and the neuron fire together while food is present, the bell synapse strengthens. Ask them to test the bell alone at a few points along the way; about seven pairings are needed from the start.",
            "reference": "Hebb (1949): when one neuron repeatedly helps fire another, the connection between them strengthens ('cells that fire together wire together' is a later paraphrase). Long-term potentiation (LTP), first reported by Bliss and Lomo in 1973 in the rabbit hippocampus, is the best-known mechanism of this kind. Many real learning rules need a third factor such as dopamine, which signals reward or surprise. In the lab the bell synapse starts at 1, grows by 1 for each pairing in which the bell, the neuron's firing and food coincide (food is fixed at 9), and falls by 0.1 each time the bell comes without food; the bell alone starts firing the neuron after 7 pairings. Pavlov described classical conditioning in dogs around 1900.",
            "minutes": 20,
        },
        {
            "title": "Forgetting and the real brain",
            "outcome": "A tested prediction of what happens when the bell stops predicting food, and a comparison of their model with a real brain.",
            "criteria": [
                "Predict how many bell-alone trials it takes for the learned response to disappear, test it, and explain the result with the learning rule.",
                "Name two ways a real brain differs from their model, and one thing the model gets right.",
            ],
            "topics": ["extinction", "memory", "brain"],
            "tools": [
                {"id": "neuron", "params": {"mode": "learning"}, "use": "Train until the bell works, then ring it alone: extinction."},
            ],
            "coaching": "Extinction: after training, ring the bell without food. Their prediction matters more than the number; if they trained far beyond seven pairings, extinction takes longer, a nice discovery to point out. Then zoom out to a real brain: 86 billion neurons, dendrites that compute, many cell types, neuromodulators, noise. Real extinction is mostly new learning layered on top rather than erasure, which is why an extinguished response can come back. Connect it to their studying: repeated retrieval strengthens memories (the Memory Palace workshop uses this).",
            "reference": "In the lab each bell-alone trial weakens the bell synapse by 0.1; straight after 7 pairings the bell stops working after about 3 bell-alone trials, and after more if they over-trained. In animals extinction is largely new inhibitory learning rather than erasure: the response can return after time (spontaneous recovery) or in a new context (renewal). The human brain has about 86 billion neurons (Azevedo et al., 2009) and on the order of 100 trillion synapses; a single neuron receives thousands of inputs on branched dendrites that do processing of their own.",
            "minutes": 15,
        },
    ],
}

MEMORY_OUTCOME = "Build a memory palace for five things you want to learn, test it for real, and make a review plan."
MEMORY_STEPS = [
    {
        "title": "Choose what you want to remember",
        "outcome": "A learning target and a familiar place for your palace.",
        "criteria": ["Name five concrete items to remember (or choose a neutral practice list).",
                     "Choose a familiar place and describe why you can mentally navigate it."],
        "topics": ["method of loci", "memory palace"],
        "tools": [],
        "coaching": "Explain the method in two or three sentences, then ask what they want to remember. Items from a course they're studying are best, because milestone 5 uses the method on real material; a neutral practice list is fine. The Simonides story is optional colour, not a question or an assessment. As soon as they give five concrete items and a familiar place they can walk through in their mind, both criteria are met: acknowledge them and complete the milestone. Do not ask for locations, imagery or recall yet; those belong to later milestones.",
        "reference": "The method of loci ('loci' means places) goes back to ancient Greek and Roman orators; Cicero tells the story of Simonides of Ceos identifying the guests at a collapsed banquet hall by remembering where each one sat. It builds on our strong memory for places and routes.",
        "minutes": 5,
    },
    {
        "title": "Make your first memorable scene",
        "outcome": "One vivid association you can explain and recall.",
        "criteria": ["Create an original, distinctive scene linking one item to a location.",
                     "Recall the item from the location cue and explain the connection in their own words."],
        "topics": ["encoding", "distinctive imagery"],
        "tools": [],
        "coaching": "Briefly explain why the item should interact with the location rather than just sit there. Model ONE unrelated example, then let them create their own. Offer non-visual options (movement, sound, a pun, a familiar sequence); not everyone visualises easily. Neuroscience vocabulary is optional context, not a gate.",
        "reference": "Images in which the item interacts with the place are remembered better than items simply placed there. Unusual images help mainly when they stand out from the others. The link between cue (the location) and item is what recall depends on, so the association has to be one the student can rebuild from the location alone.",
        "minutes": 8,
    },
    {
        "title": "Build your five-stop route",
        "outcome": "An ordered route with five locations and five student-created associations.",
        "criteria": ["Describe five distinct locations in a consistent order.",
                     "Link each target item to its own location with a distinctive association."],
        "topics": ["route", "ordered recall"],
        "tools": [],
        "coaching": "Reuse their earlier target and scene. Build one location at a time if needed. They supply the route and the associations; don't accept a complete palace written by Pedro. Check for locations that look alike or cues that overlap. There is no single right scene: judge whether each association is usable.",
        "reference": "Distinct, well-separated locations along a route they know well avoid interference between items. The route's fixed order is what lets the palace hold ordered lists (a speech, the steps of a process).",
        "minutes": 12,
    },
    {
        "title": "Walk, recall, and repair",
        "outcome": "A recall test of all five items with the answers hidden, and stronger cues for anything missed.",
        "criteria": ["Set up the route in the recall test and recall all five items in route order, with the items hidden and without Pedro displaying the answers.",
                     "Repair any missed associations and make a fresh unaided attempt that retrieves all five."],
        "topics": ["retrieval practice", "recall"],
        "tools": [
            {"id": "recall", "params": {}, "use": "At the start of the milestone: they enter their five places and items, then take the test with the items hidden and send the result."},
        ],
        "coaching": "Use the recall test: they enter their route once, then the lab hides the items while they recall and scores each place. Do not print the targets or reveal missing items in the chat before their attempt. Do not infer success from 'done' or 'I got them': judge from the lab's result. For a missed item, work out with them why the cue failed and strengthen that one association; hints are practice, so follow them with a new unaided attempt. The lab can't stop them looking elsewhere: describe it as observed practice, not proof they did not look. Ten items is an optional extension, never a requirement.",
        "reference": "Retrieving from memory (testing yourself) strengthens memories more than rereading does (the testing effect). A missed item usually means a weak or ambiguous cue at that location, so the fix is a stronger link, not more repetition of the list.",
        "minutes": 10,
    },
    {
        "title": "Use it in your own studying",
        "outcome": "A reusable study plan and an independently explained example from your course.",
        "criteria": ["Apply the method to a real study item and explain its meaning, not just its mnemonic.",
                     "Choose when to attempt delayed recall and explain one situation where understanding or problem solving is also needed."],
        "topics": ["spaced repetition", "study plan"],
        "tools": [
            {"id": "recall", "params": {}, "use": "Optional: a delayed test of the same route in a later session, to see how much stayed."},
        ],
        "coaching": "If they used a practice list, transfer one item to material they actually study. Never claim mnemonics replace understanding or guarantee retention. Suggest a delayed recall test in the lab a day and a few days later. At the end summarise only the target, route, associations and recall they actually demonstrated, and invite them to save that summary to their notes. Don't say a reminder was scheduled or notes were saved automatically.",
        "reference": "In a 2017 study (Dresler et al., Neuron), people who practised the method of loci for about six weeks went from recalling about 26 to about 62 words of a 72-word list, and still did better four months later. Mnemonics help with facts and orders; they don't create understanding. Spaced retrieval (for example a day, a few days and a week later) keeps memories far longer than cramming.",
        "minutes": 10,
    },
]
LEGACY_MEMORY_TITLES = ["The Origin of the Method", "Your Brain's Spatial Hardware", "Building & Encoding",
                        "Hands-On Practice", "From Champions to Your Exams"]
MEMORY = {"outcome": MEMORY_OUTCOME, "steps": MEMORY_STEPS, "legacy_titles": LEGACY_MEMORY_TITLES}

# Folder name → workshop. Folder names are stored in students' outlines: never rename them.
LIBRARY = {
    "Build Your Own LLM": LLM,
    "Build a Rocket": ROCKET,
    "Build a Brain": BRAIN,
    "Memory Palace": MEMORY,
}

# Workshops that teach from their own contracts and labs, with no PDF sources to index.
WITHOUT_SOURCES = {"Build Your Own LLM", "Build a Rocket", "Build a Brain"}
