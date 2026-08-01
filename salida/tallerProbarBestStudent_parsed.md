# tallerProbarBestStudent

## Párrafos

**[section_header]** Mealy and Moore Machines

**[text]** Camilo Andres Ni˜ no Amaya

**[text]** Systems and Computer Engineering

**[text]** UPTC

**[text]** Sogamoso, Colombia camilo.nino02@uptc.edu.co

**[text]** Abstract -This paper presents a concise study of two classical finite-state machine models with output: the Mealy machine and the Moore machine. The report answers theoretical questions covering the conceptual description, formal definitions, and main applications of both models. The fourth question, corresponding to a Mealy-machine case study, is intentionally omitted according to the assignment instructions. The discussion emphasizes the central difference between both models: in a Mealy machine, the output depends on the present state and the present input, whereas in a Moore machine, the output depends only on the present state. The document concludes with a case study of a Moore machine, a comparative analysis, and practical construction details using JFLAP.

**[text]** Index Terms -Finite-state machine, Mealy machine, Moore machine, sequential circuit, finite-state transducer, JFLAP.

**[section_header]** I. INTRODUCTION

**[text]** Finite-state machines (FSMs) are formal models for systems whose behavior can be represented by a finite set of internal states, input symbols, output symbols, and transition rules. When an FSM produces output symbols, it is commonly understood as a finite-state transducer rather than only as an acceptor of strings [1], [2]. Two classical output models are widely used in automata theory and digital design: the Mealy model, historically associated with Mealy's synthesis method for sequential circuits [3], and the Moore model, associated with Moore's work on sequential machines [4].

**[text]** The distinction is relevant because both models can describe input-output behavior, but they associate outputs with different parts of the machine. In a Mealy machine, outputs are associated with transitions because the output function depends on the current state and input. In a Moore machine, outputs are associated with states because the output function depends only on the current state [5], [6].

**[section_header]** II. QUESTION 1: DESCRIPTION OF A MEALY MACHINE

**[text]** A Mealy machine is a finite-state machine with output in which the output produced at a given step is determined by two elements: the present internal state and the present input symbol. In digital-system notation, this idea is often written as

**[text]** where S ( t ) is the present state, X ( t ) is the present input, and y ( t ) is the current output [5], [6]. Therefore, a Mealy machine behaves as a deterministic transducer that maps an input sequence into an output sequence.

**[text]** In a state-transition diagram, a Mealy machine is usually represented by directed edges labeled with pairs of the form a/b , where a is the input symbol that enables the transition and b is the output symbol produced during that transition. This means that the output is naturally associated with the transition, not exclusively with the destination state.

**[section_header]** III. QUESTION 2: FORMAL DEFINITION OF A MEALY MACHINE

**[text]** A deterministic Mealy machine can be formally defined as the 6-tuple

**[text]** where:

**[list_item]** Q is a finite, nonempty set of states;

**[list_item]** Σ is a finite input alphabet;

**[list_item]** Γ is a finite output alphabet;

**[list_item]** δ : Q × Σ → Q is the state-transition function;

**[list_item]** λ : Q × Σ → Γ is the output function;

**[list_item]** q 0 ∈ Q is the initial state.

**[text]** This formalization captures the defining property of the Mealy model: the output function has domain Q × Σ , so an output symbol depends on both a state and the input being processed [1], [3].

**[section_header]** IV. QUESTION 3: APPLICATIONS OF MEALY MACHINES

**[text]** Mealy machines are useful when the output of a system must depend on both the system condition and the input currently being received.

**[list_item]** Sequential digital controllers: Used to synthesize sequential circuits whose control signals depend on input events and the current state [5], [7].

**[list_item]** Sequence detectors: Convenient for detecting input patterns in serial streams, generating output on the exact transition where the final pattern symbol is read.

**[list_item]** Communication protocols: Modeling logic where output actions (e.g., error flags) depend on both state and the most recent event.

**[list_item]** Lexical analysis: Widely used as transducers in language processing to transform strings into output strings [2].

**[section_header]** V. QUESTION 5: DESCRIPTION OF A MOORE MACHINE

**[text]** A Moore machine is a finite-state machine with output in which the output depends only on the present state. In digitalsystem notation, this is expressed as

**[text]** which contrasts with the Mealy equation [5], [6]. In a Mooremachine state diagram, each state is labeled with its associated output, and edges are labeled only by input conditions. When the machine enters a state, the output assigned to that state becomes the machine output. This makes Moore machines safer against transient input glitches in synchronous hardware.

**[section_header]** VI. QUESTION 6: FORMAL DEFINITION OF A MOORE MACHINE

**[text]** A deterministic Moore machine is defined as the 6-tuple

**[text]** where:

**[list_item]** Q , Σ , Γ , δ , and q 0 are defined exactly as in the Mealy machine;

**[list_item]** λ : Q → Γ is the output function.

**[text]** The essential formal difference from the Mealy definition is the domain of the output function: λ maps states directly to output symbols [1], [4].

**[caption]** TABLE I FORMAL DIFFERENCE BETWEEN MEALY AND MOORE MACHINES

**[section_header]** VII. QUESTION 7: APPLICATIONS OF MOORE MACHINES

**[text]** Because Moore machines isolate their outputs from direct combinational paths linked to inputs, they are favored in systems requiring highly stable, synchronized signals [5]. Key applications include:

**[list_item]** Traffic Light Controllers: The state of the traffic light purely dictates the output signals sent to the bulbs. Inputs (like pedestrian buttons or car sensors) only influence the next state transition, not the direct output, preventing dangerous mid-cycle flickering.

**[list_item]** Digital Counters and Clock Dividers: The output of a digital counter is structurally identical to its current state.

**[list_item]** SRAM/Memory Controllers: Hardware control logic where timing is critical, and outputs (like Write Enable or Read Enable) must be glitch-free and tied strictly to a stable clock edge.

**[section_header]** VIII. QUESTION 8: CASE STUDY OF A MOORE MACHINE

**[text]** Consider a Binary Parity Checker designed to output '1' if the number of '1's received in a serial bitstream is odd, and '0' if the number is even.

**[text]** In a Moore machine implementation, the outputs are intrinsic to the states:

**[list_item]** States ( Q ): S even (Even number of 1s), S odd (Odd number of 1s).

**[list_item]** Outputs ( λ ): λ ( S even ) = 0 , λ ( S odd ) = 1 .

**[list_item]** Transitions ( δ ):

**[list_item]** -From S even : Input '0' → S even ; Input '1' → S odd .

**[list_item]** -From S odd : Input '0' → S odd ; Input '1' → S even .

**[text]** In this design, the current parity is physically stored as the state. The output simply reflects which state the machine is currently occupying, completely independent of whatever the next incoming bit might be before the clock cycle completes.

**[section_header]** IX. QUESTION 9: DIFFERENCES AND SIMILARITIES

**[text]** Both Mealy and Moore machines share several fundamental similarities :

**[list_item]** Both are finite-state transducers utilized to model sequential logic behavior.

**[list_item]** Both possess identical computational power. According to automata theory, any regular language or transduction computable by a Mealy machine can be computed by an equivalent Moore machine, and vice versa [1].

**[list_item]** Both mathematically require a finite set of states, a starting state, and input/output alphabets.

**[text]** However, their structural distinctions create important differences :

**[list_item]** Output Dependency: Mealy outputs depend on Q × Σ (State and Input), whereas Moore outputs depend entirely on Q (State alone).

**[list_item]** State Count: A Mealy machine typically requires fewer states than an equivalent Moore machine, because a single state in a Mealy model can emit multiple different outputs depending on the transition [5].

**[list_item]** Response Delay: A Mealy machine can react to an input change instantly (asynchronously within the same clock cycle). A Moore machine inherently introduces a onecycle delay, as the machine must first transition into a new state before the output updates.

**[list_item]** Hardware Stability: Moore machines are inherently immune to input combinational glitches, making them preferable when clean output signals are strictly required.

**[section_header]** X. QUESTION 10: CONSTRUCTION IN JFLAP

**[text]** JFLAP (Java Formal Languages and Automata Package) provides a graphical environment to construct and simulate finite-state transducers. To fulfill this requirement, two basic machines were designed. The following state transition tables provide the exact parameters used for their construction in the software.

**[section_header]** A. Mealy Machine in JFLAP

**[text]** The selected Mealy machine is a Sequence Detector that outputs a '1' whenever the sequence '11' is detected in a binary stream, and '0' otherwise. In JFLAP, when drawing a transition between states, the software prompts for the input and output in an input;output format on the edges.

**[caption]** TABLE II TRANSITION TABLE FOR MEALY MACHINE ('11' SEQUENCE DETECTOR)

**[text]** The final implementation constructed in JFLAP is shown in Fig. 1.

**[caption]** Fig. 1. Mealy Machine implementation in JFLAP.

**[text]** q1

**[text]** 0;0

**[text]** 1:0

**[text]** 0;0

**[text]** q0

**[section_header]** B. Moore Machine in JFLAP

**[text]** The selected Moore machine is the Odd Parity Checker described in the previous case study. In JFLAP, the output is assigned directly to the state, visually rendering the output in a box attached to the state circle. The edges only carry the input symbol.

**[caption]** TABLE III TRANSITION TABLE FOR MOORE MACHINE (ODD PARITY CHECKER)

**[text]** The final implementation constructed in JFLAP is shown in Fig. 2.

**[section_header]** XI. CONCLUSION

**[text]** Mealy and Moore machines are two foundational models of finite-state machines with output. A Mealy machine computes its output from the current state and the current input, so it is well suited for systems that must react immediately to input events and for minimizing state counts. A Moore machine computes its output only from the current state, so it is often easier to reason about in synchronous digital designs and protects against input-driven glitches. Their main formal distinction is the mathematical domain of the output function. This theoretical difference practically dictates how outputs are represented in formal diagrams like JFLAP, how outputs are timed in hardware circuits, and how complex the control logic becomes.

**[caption]** Fig. 2. Moore Machine implementation in JFLAP.

**[text]** 0

**[text]** q0

**[text]** q1

**[section_header]** REFERENCES

**[list_item]** J. E. Hopcroft, R. Motwani, and J. D. Ullman, Introduction to Automata Theory, Languages, and Computation , 3rd ed. Boston, MA: AddisonWesley, 2006.

**[list_item]** M. Mohri, 'Finite-state transducers in language and speech processing,' Computational Linguistics , vol. 23, no. 2, pp. 269-311, 1997. [Online]. Available: https://aclanthology.org/J97-2003/

**[list_item]** G. H. Mealy, 'A method for synthesizing sequential circuits,' The Bell System Technical Journal , vol. 34, no. 5, pp. 1045-1079, Sep. 1955.

**[list_item]** E. F. Moore, 'Gedanken-experiments on sequential machines,' in Automata Studies , ser. Annals of Mathematics Studies, C. E. Shannon and J. McCarthy, Eds. Princeton, NJ: Princeton University Press, 1956, no. 34, pp. 129-153, reprinted by Princeton University Press/De Gruyter with DOI: 10.1515/9781400882618-006.

**[list_item]** P. P. Chu, RTL Hardware Design Using VHDL: Coding for Efficiency, Portability, and Scalability . Hoboken, NJ: John Wiley & Sons, 2006, chapter 10 discusses finite-state machine principles and practice.

**[list_item]** D. Mirza, 'Lecture 14: Sequential networks -finite state machines: Moore and mealy,' CSE 140, University of California, San Diego, 2015. [Online]. Available: https://cseweb.ucsd.edu/classes/wi15/ cse140-ab/slides/lec14 before.pdf

**[list_item]** Cornell University Department of Computer Science, 'Finite state machines,' CS 3410 Computer System Organization and Programming, lecture slides, 2019. [Online]. Available: https://www.cs.cornell.edu/ courses/cs3410/2019sp/schedule/slides/05-fsm-notes.pdf

## Chunks Semánticos

### Chunk 1 (1 oraciones)

Mealy and Moore Machines

### Chunk 2 (1 oraciones)

Camilo Andres Ni˜ no Amaya

### Chunk 3 (1 oraciones)

Systems and Computer Engineering

### Chunk 4 (1 oraciones)

UPTC

### Chunk 5 (1 oraciones)

Sogamoso, Colombia camilo.nino02@uptc.edu.co

### Chunk 6 (1 oraciones)

Abstract -This paper presents a concise study of two classical finite-state machine models with output: the Mealy machine and the Moore machine.

### Chunk 7 (1 oraciones)

The report answers theoretical questions covering the conceptual description, formal definitions, and main applications of both models.

### Chunk 8 (1 oraciones)

The fourth question, corresponding to a Mealy-machine case study, is intentionally omitted according to the assignment instructions.

### Chunk 9 (1 oraciones)

The discussion emphasizes the central difference between both models: in a Mealy machine, the output depends on the present state and the present input, whereas in a Moore machine, the output depends only on the present state.

### Chunk 10 (1 oraciones)

The document concludes with a case study of a Moore machine, a comparative analysis, and practical construction details using JFLAP.

### Chunk 11 (1 oraciones)

Index Terms -Finite-state machine, Mealy machine, Moore machine, sequential circuit, finite-state transducer, JFLAP.

### Chunk 12 (1 oraciones)

I. INTRODUCTION

### Chunk 13 (2 oraciones)

Finite-state machines (FSMs) are formal models for systems whose behavior can be represented by a finite set of internal states, input symbols, output symbols, and transition rules. When an FSM produces output symbols, it is commonly understood as a finite-state transducer rather than only as an acceptor of strings [1], [2].

### Chunk 14 (1 oraciones)

Two classical output models are widely used in automata theory and digital design: the Mealy model, historically associated with Mealy's synthesis method for sequential circuits [3], and the Moore model, associated with Moore's work on sequential machines [4].

### Chunk 15 (1 oraciones)

The distinction is relevant because both models can describe input-output behavior, but they associate outputs with different parts of the machine.

### Chunk 16 (1 oraciones)

In a Mealy machine, outputs are associated with transitions because the output function depends on the current state and input.

### Chunk 17 (1 oraciones)

In a Moore machine, outputs are associated with states because the output function depends only on the current state [5], [6].

### Chunk 18 (1 oraciones)

II.

### Chunk 19 (2 oraciones)

QUESTION 1: DESCRIPTION OF A MEALY MACHINE A Mealy machine is a finite-state machine with output in which the output produced at a given step is determined by two elements: the present internal state and the present input symbol.

### Chunk 20 (1 oraciones)

In digital-system notation, this idea is often written as

### Chunk 21 (1 oraciones)

where S ( t ) is the present state, X ( t ) is the present input, and y ( t ) is the current output [5], [6].

### Chunk 22 (2 oraciones)

Therefore, a Mealy machine behaves as a deterministic transducer that maps an input sequence into an output sequence. In a state-transition diagram, a Mealy machine is usually represented by directed edges labeled with pairs of the form a/b , where a is the input symbol that enables the transition and b is the output symbol produced during that transition.

### Chunk 23 (1 oraciones)

This means that the output is naturally associated with the transition, not exclusively with the destination state.

### Chunk 24 (1 oraciones)

III.

### Chunk 25 (2 oraciones)

QUESTION 2: FORMAL DEFINITION OF A MEALY MACHINE A deterministic Mealy machine can be formally defined as the 6-tuple

### Chunk 26 (1 oraciones)

where:

### Chunk 27 (1 oraciones)

Q is a finite, nonempty set of states;

### Chunk 28 (1 oraciones)

Σ is a finite input alphabet;

### Chunk 29 (1 oraciones)

Γ is a finite output alphabet;

### Chunk 30 (1 oraciones)

δ : Q × Σ → Q is the state-transition function;

### Chunk 31 (1 oraciones)

λ : Q × Σ → Γ is the output function;

### Chunk 32 (1 oraciones)

q 0 ∈ Q is the initial state.

### Chunk 33 (1 oraciones)

This formalization captures the defining property of the Mealy model: the output function has domain Q × Σ , so an output symbol depends on both a state and the input being processed [1], [3].

### Chunk 34 (1 oraciones)

IV.

### Chunk 35 (2 oraciones)

QUESTION 3: APPLICATIONS OF MEALY MACHINES Mealy machines are useful when the output of a system must depend on both the system condition and the input currently being received.

### Chunk 36 (1 oraciones)

Sequential digital controllers: Used to synthesize sequential circuits whose control signals depend on input events and the current state [5], [7].

### Chunk 37 (1 oraciones)

Sequence detectors: Convenient for detecting input patterns in serial streams, generating output on the exact transition where the final pattern symbol is read.

### Chunk 38 (1 oraciones)

Communication protocols: Modeling logic where output actions (e.g., error flags) depend on both state and the most recent event.

### Chunk 39 (1 oraciones)

Lexical analysis: Widely used as transducers in language processing to transform strings into output strings [2].

### Chunk 40 (2 oraciones)

V. QUESTION 5: DESCRIPTION OF A MOORE MACHINE A Moore machine is a finite-state machine with output in which the output depends only on the present state.

### Chunk 41 (1 oraciones)

In digitalsystem notation, this is expressed as

### Chunk 42 (1 oraciones)

which contrasts with the Mealy equation [5], [6].

### Chunk 43 (1 oraciones)

In a Mooremachine state diagram, each state is labeled with its associated output, and edges are labeled only by input conditions.

### Chunk 44 (1 oraciones)

When the machine enters a state, the output assigned to that state becomes the machine output.

### Chunk 45 (1 oraciones)

This makes Moore machines safer against transient input glitches in synchronous hardware.

### Chunk 46 (1 oraciones)

VI.

### Chunk 47 (2 oraciones)

QUESTION 6: FORMAL DEFINITION OF A MOORE MACHINE A deterministic Moore machine is defined as the 6-tuple

### Chunk 48 (1 oraciones)

where:

### Chunk 49 (1 oraciones)

Q , Σ , Γ , δ , and q 0 are defined exactly as in the Mealy machine;

### Chunk 50 (1 oraciones)

λ : Q → Γ is the output function.

### Chunk 51 (1 oraciones)

The essential formal difference from the Mealy definition is the domain of the output function: λ maps states directly to output symbols [1], [4].

### Chunk 52 (1 oraciones)

TABLE I FORMAL DIFFERENCE BETWEEN MEALY AND MOORE MACHINES

### Chunk 53 (1 oraciones)

VII.

### Chunk 54 (1 oraciones)

QUESTION 7: APPLICATIONS OF MOORE MACHINES

### Chunk 55 (1 oraciones)

Because Moore machines isolate their outputs from direct combinational paths linked to inputs, they are favored in systems requiring highly stable, synchronized signals [5].

### Chunk 56 (1 oraciones)

Key applications include:

### Chunk 57 (1 oraciones)

Traffic Light Controllers: The state of the traffic light purely dictates the output signals sent to the bulbs.

### Chunk 58 (1 oraciones)

Inputs (like pedestrian buttons or car sensors) only influence the next state transition, not the direct output, preventing dangerous mid-cycle flickering.

### Chunk 59 (1 oraciones)

Digital Counters and Clock Dividers: The output of a digital counter is structurally identical to its current state.

### Chunk 60 (1 oraciones)

SRAM/Memory Controllers: Hardware control logic where timing is critical, and outputs (like Write Enable or Read Enable) must be glitch-free and tied strictly to a stable clock edge.

### Chunk 61 (1 oraciones)

VIII.

### Chunk 62 (1 oraciones)

QUESTION 8: CASE STUDY OF A MOORE MACHINE

### Chunk 63 (1 oraciones)

Consider a Binary Parity Checker designed to output '1' if the number of '1's received in a serial bitstream is odd, and '0' if the number is even.

### Chunk 64 (1 oraciones)

In a Moore machine implementation, the outputs are intrinsic to the states:

### Chunk 65 (1 oraciones)

States ( Q ): S even (Even number of 1s), S odd (Odd number of 1s).

### Chunk 66 (1 oraciones)

Outputs ( λ ): λ ( S even ) = 0 , λ ( S odd ) = 1 .

### Chunk 67 (1 oraciones)

Transitions ( δ ):

### Chunk 68 (2 oraciones)

-From S even : Input '0' → S even ; Input '1' → S odd . -From S odd : Input '0' → S odd ; Input '1' → S even .

### Chunk 69 (1 oraciones)

In this design, the current parity is physically stored as the state.

### Chunk 70 (1 oraciones)

The output simply reflects which state the machine is currently occupying, completely independent of whatever the next incoming bit might be before the clock cycle completes.

### Chunk 71 (1 oraciones)

IX.

### Chunk 72 (1 oraciones)

QUESTION 9: DIFFERENCES AND SIMILARITIES

### Chunk 73 (1 oraciones)

Both Mealy and Moore machines share several fundamental similarities :

### Chunk 74 (1 oraciones)

Both are finite-state transducers utilized to model sequential logic behavior.

### Chunk 75 (1 oraciones)

Both possess identical computational power.

### Chunk 76 (1 oraciones)

According to automata theory, any regular language or transduction computable by a Mealy machine can be computed by an equivalent Moore machine, and vice versa [1].

### Chunk 77 (1 oraciones)

Both mathematically require a finite set of states, a starting state, and input/output alphabets.

### Chunk 78 (1 oraciones)

However, their structural distinctions create important differences :

### Chunk 79 (2 oraciones)

Output Dependency: Mealy outputs depend on Q × Σ (State and Input), whereas Moore outputs depend entirely on Q (State alone). State Count: A Mealy machine typically requires fewer states than an equivalent Moore machine, because a single state in a Mealy model can emit multiple different outputs depending on the transition [5].

### Chunk 80 (1 oraciones)

Response Delay: A Mealy machine can react to an input change instantly (asynchronously within the same clock cycle).

### Chunk 81 (1 oraciones)

A Moore machine inherently introduces a onecycle delay, as the machine must first transition into a new state before the output updates.

### Chunk 82 (1 oraciones)

Hardware Stability: Moore machines are inherently immune to input combinational glitches, making them preferable when clean output signals are strictly required.

### Chunk 83 (1 oraciones)

X. QUESTION 10: CONSTRUCTION IN JFLAP

### Chunk 84 (1 oraciones)

JFLAP (Java Formal Languages and Automata Package) provides a graphical environment to construct and simulate finite-state transducers.

### Chunk 85 (1 oraciones)

To fulfill this requirement, two basic machines were designed.

### Chunk 86 (1 oraciones)

The following state transition tables provide the exact parameters used for their construction in the software.

### Chunk 87 (1 oraciones)

A. Mealy Machine in JFLAP

### Chunk 88 (1 oraciones)

The selected Mealy machine is a Sequence Detector that outputs a '1' whenever the sequence '11' is detected in a binary stream, and '0' otherwise.

### Chunk 89 (1 oraciones)

In JFLAP, when drawing a transition between states, the software prompts for the input and output in an input;output format on the edges.

### Chunk 90 (1 oraciones)

TABLE II TRANSITION TABLE FOR MEALY MACHINE ('11' SEQUENCE DETECTOR)

### Chunk 91 (1 oraciones)

The final implementation constructed in JFLAP is shown in Fig.

### Chunk 92 (1 oraciones)

1.

### Chunk 93 (1 oraciones)

Fig.

### Chunk 94 (1 oraciones)

1.

### Chunk 95 (1 oraciones)

Mealy Machine implementation in JFLAP.

### Chunk 96 (1 oraciones)

q1

### Chunk 97 (3 oraciones)

0;0 1:0 0;0

### Chunk 98 (1 oraciones)

q0

### Chunk 99 (1 oraciones)

B. Moore Machine in JFLAP

### Chunk 100 (1 oraciones)

The selected Moore machine is the Odd Parity Checker described in the previous case study.

### Chunk 101 (1 oraciones)

In JFLAP, the output is assigned directly to the state, visually rendering the output in a box attached to the state circle.

### Chunk 102 (1 oraciones)

The edges only carry the input symbol.

### Chunk 103 (1 oraciones)

TABLE III TRANSITION TABLE FOR MOORE MACHINE (ODD PARITY CHECKER)

### Chunk 104 (1 oraciones)

The final implementation constructed in JFLAP is shown in Fig.

### Chunk 105 (1 oraciones)

2.

### Chunk 106 (1 oraciones)

XI.

### Chunk 107 (1 oraciones)

CONCLUSION

### Chunk 108 (2 oraciones)

Mealy and Moore machines are two foundational models of finite-state machines with output. A Mealy machine computes its output from the current state and the current input, so it is well suited for systems that must react immediately to input events and for minimizing state counts.

### Chunk 109 (1 oraciones)

A Moore machine computes its output only from the current state, so it is often easier to reason about in synchronous digital designs and protects against input-driven glitches.

### Chunk 110 (1 oraciones)

Their main formal distinction is the mathematical domain of the output function.

### Chunk 111 (1 oraciones)

This theoretical difference practically dictates how outputs are represented in formal diagrams like JFLAP, how outputs are timed in hardware circuits, and how complex the control logic becomes.

### Chunk 112 (1 oraciones)

Fig.

### Chunk 113 (1 oraciones)

2.

### Chunk 114 (1 oraciones)

Moore Machine implementation in JFLAP.

### Chunk 115 (1 oraciones)

0

### Chunk 116 (2 oraciones)

q0 q1

### Chunk 117 (1 oraciones)

REFERENCES

### Chunk 118 (1 oraciones)

J. E. Hopcroft, R. Motwani, and J. D. Ullman, Introduction to Automata Theory, Languages, and Computation , 3rd ed.

### Chunk 119 (1 oraciones)

Boston, MA: AddisonWesley, 2006.

### Chunk 120 (1 oraciones)

M. Mohri, 'Finite-state transducers in language and speech processing,' Computational Linguistics , vol.

### Chunk 121 (1 oraciones)

23, no.

### Chunk 122 (1 oraciones)

2, pp.

### Chunk 123 (1 oraciones)

269-311, 1997. [

### Chunk 124 (1 oraciones)

Online].

### Chunk 125 (1 oraciones)

Available: https://aclanthology.org/J97-2003/

### Chunk 126 (1 oraciones)

G. H. Mealy, 'A method for synthesizing sequential circuits,' The Bell System Technical Journal , vol.

### Chunk 127 (1 oraciones)

34, no.

### Chunk 128 (1 oraciones)

5, pp.

### Chunk 129 (1 oraciones)

1045-1079, Sep. 1955.

### Chunk 130 (1 oraciones)

E. F. Moore, 'Gedanken-experiments on sequential machines,' in Automata Studies , ser.

### Chunk 131 (1 oraciones)

Annals of Mathematics Studies, C. E. Shannon and J. McCarthy, Eds.

### Chunk 132 (1 oraciones)

Princeton, NJ: Princeton University Press, 1956, no.

### Chunk 133 (1 oraciones)

34, pp.

### Chunk 134 (1 oraciones)

129-153, reprinted by Princeton University Press/De Gruyter with DOI: 10.1515/9781400882618-006.

### Chunk 135 (1 oraciones)

P. P. Chu, RTL Hardware Design Using VHDL: Coding for Efficiency, Portability, and Scalability .

### Chunk 136 (1 oraciones)

Hoboken, NJ: John Wiley & Sons, 2006, chapter 10 discusses finite-state machine principles and practice.

### Chunk 137 (1 oraciones)

D. Mirza, 'Lecture 14: Sequential networks -finite state machines: Moore and mealy,' CSE 140, University of California, San Diego, 2015. [

### Chunk 138 (1 oraciones)

Online].

### Chunk 139 (1 oraciones)

Available: https://cseweb.ucsd.edu/classes/wi15/ cse140-ab/slides/lec14 before.pdf

### Chunk 140 (1 oraciones)

Cornell University Department of Computer Science, 'Finite state machines,' CS 3410 Computer System Organization and Programming, lecture slides, 2019. [

### Chunk 141 (1 oraciones)

Online].

### Chunk 142 (1 oraciones)

Available: https://www.cs.cornell.edu/ courses/cs3410/2019sp/schedule/slides/05-fsm-notes.pdf

## Tablas

### Tabla 1

| Model | Output function | Output associated with |
| --- | --- | --- |
| Mealy Moore | λ : Q × Σ → Γ λ : Q → Γ | Transition State |

### Tabla 2

| Present State. | Next State.Input = 0 | Next State.Input = 1 | Output.Input = 0 | Output.Input = |
| --- | --- | --- | --- | --- |
| q 0 (Initial) q 1 | q 0 q 0 | q 1 q 1 | 0 0 | 0 1 |

### Tabla 3

| Present State. | Next State.Input = 0 | Next State.Input = 1 | Output. |
| --- | --- | --- | --- |
| q 0 (Even parity) q 1 (Odd parity) | q 0 q 1 | q 1 q 0 | 0 1 |

## Figuras

### Figura 1

### Figura 2
