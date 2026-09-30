"""Regression cases for Pedro's tutoring, built from the Network Science lesson audit (28 Sep 2026).

Each case replays a real moment of the test account's lesson (user 11) and checks the reply
against verified answers (L02 physical PDF pages; values recomputed independently).
`until` replays a stored student message with the history before it; `section`+`message`
asks a fresh question inside that section. Sections follow the roadmap regenerated on 29 Sep 2026
(page-exact sections); `section_title` guards against the roadmap changing again. `check` lines are an evaluation-only request so
numeric answers can be graded by code rather than by opinion.
"""

USER = 11
FOLDER = "Network Science"

CASES = [
    {
        "id": "s10-opener",
        "section": 13,
        "section_title": "Shortest Paths",
        "why": "Section 10 opener: must not misread the page 25 matrix; should use the relevant diagrams.",
        "until": 2411,
        "grade": "opener",
    },
    {
        "id": "s10-matrix-read",
        "section": 13,
        "section_title": "Shortest Paths",
        "why": "Reading the adjacency matrix and its powers on L02 pp. 25-27.",
        "message": "Quick check before we go on: on the matrix on page 25, is there a direct edge between node 9 and node 10? "
                   "What are the entries A²(9,10) and A³(9,10) on pages 26-27, and what is the shortest distance d(9,10)?",
        "check": "CHECK: edge=yes|no; A2=<integer>; A3=<integer>; d=<integer>",
        "expect": {"edge": "yes", "A2": 0, "A3": 2, "d": 1},
    },
    {
        "id": "s10-student-challenge",
        "section": 13,
        "section_title": "Shortest Paths",
        "why": "The student correctly reads the 9-10 edge after Pedro wrongly said there is none.",
        "until": 2413,
        "grade": "tutor_error",
        "facts": "A(9,10) = 1, so nodes 9 and 10 are directly connected and d(9,10) = 1. A²(9,10) = 0 and "
                 "A³(9,10) = 2 (walks 9-8-9-10 and 9-10-9-10; walks may revisit nodes). The student's reading "
                 "(there is an edge; 3-step walks can backtrack) is correct; Pedro's earlier claim that 9 and 10 "
                 "are not directly connected was wrong.",
    },
    {
        "id": "s17-tutor-induced",
        "why": "Pedro's opener put randomization under the evolution pillar; the student repeats it. Rebuilt for "
               "the regenerated roadmap: Pedro's original opener (message 2498), minus its 'final section' framing, "
               "inside the section that now holds the two-pillar slide, then the student's real answer (2499).",
        "section": 1,
        "section_title": "Course Objectives",
        "history": [
            ["user", 'I\'m ready to learn about "Complex Systems and Course Objectives". Please teach me this section.'],
            ["pedro", "Before we get into the graph theory, let's look at how the course frames itself. The lecture "
                      "([L1 1 Kickoff · p. 34](#lesson-source/src_9afa1442e4/34)) says the whole course is built around "
                      "**two pillars**:\n\n1. **Methods to analyse networks**: the toolbox. Graph theory basics, node degree "
                      "and degree distributions, clustering coefficient, assortativity, paths and distances, connectivity, "
                      "statistical properties and visualization. All of these are ways of *measuring and describing* a "
                      "network you already have.\n2. **Understanding the mechanisms that lead to their evolution**: the "
                      "\"why\" side. Network randomization techniques (to test whether a pattern is meaningful or just "
                      "chance), complex systems and emergence (how simple local rules produce global structure), and "
                      "applications such as rumour spreading and financial networks.\n\n**Quick check:** if I gave you the "
                      "clustering coefficient of a real network and asked \"is this high clustering meaningful, or could it "
                      "happen by chance?\", which pillar would you use, which specific tool from that pillar would you "
                      "apply, and why?"],
        ],
        "message": "I would say Understanding the mechanisms that lead to their evolution specifically network "
                   "randomization techniques",
        "grade": "tutor_error",
        "facts": "Network randomization is an analysis method (pillar 1: methods to analyse networks; the course "
                 "objectives on Kickoff p. 15 list measuring features and discerning whether they are significant). "
                 "Pedro's own opening message listed it under pillar 2, and the student followed that. The student's "
                 "answer reflects Pedro's error, not an independent misconception.",
    },
    {
        "id": "s6-reciprocity",
        "section": 15,
        "section_title": "Reciprocity",
        "why": "Reciprocity counts both directed edges of every mutual pair (L02 pp. 39-42).",
        "message": "A directed network has 8 directed links that form exactly 4 mutual pairs (every link has its reverse). "
                   "What is its reciprocity R? And if instead it had 2 mutual pairs plus 4 one-way links, what would R be?",
        "check": "CHECK: R1=<number>; R2=<number>",
        "expect": {"R1": 1.0, "R2": 0.5},
    },
    {
        "id": "s16-density",
        "section": 10,
        "section_title": "Histograms",
        "why": "Probability density of a histogram bin needs the total count (L01 p. 103).",
        "message": "For a histogram bin i that contains c_i of the M observations and has width Δ_i, what is the "
                   "estimated probability density p_i? Give the formula.",
        "check": "CHECK: p_i=<formula using c_i, M and Δ_i>",
        "grade": "density",
    },
    {
        "id": "s7-mean-clustering",
        "section": 16,
        "section_title": "Clustering Coefficient",
        "why": "Mean local clustering of the lecture's example network is 13/25 = 0.52.",
        "message": "What is the average local clustering coefficient of the example network in the lecture slides? "
                   "List each node's value and the mean.",
        "check": "CHECK: mean=<number>",
        "expect": {"mean": 0.52},
    },
    {
        "id": "s5-valid-exercise",
        "section": 8,
        "section_title": "Node Degree",
        "why": "Generated exercises must be valid simple graphs (degree <= N-1, degrees match edges).",
        "message": "Can you make me a new practice exercise: a simple undirected graph with 5 nodes numbered 1-5? "
                   "Give its edge list and each node's degree, then ask me something about it.",
        "check": "CHECK: edges=<a-b,c-d,...>; degrees=<d1,d2,d3,d4,d5>",
        "grade": "graph",
    },
    {
        "id": "s9-null-model",
        "section": 20,
        "section_title": "Significance",
        "why": "A property that survives degree-preserving randomization is explained by the degree sequence.",
        "message": "If a network's clustering coefficient stays about the same after degree-preserving randomization, "
                   "does that suggest the clustering reflects structure beyond the degree sequence?",
        "check": "CHECK: beyond_degrees=yes|no",
        "expect": {"beyond_degrees": "no"},
    },
    {
        "id": "s14-small-world",
        "section": 3,
        "section_title": "Degrees of Separation",
        "why": "Watts-Strogatz small world: high clustering with path lengths comparable to a random graph.",
        "message": "In the Watts-Strogatz sense, how does a small-world network compare with a random network of the "
                   "same size and density, in clustering and in average path length?",
        "check": "CHECK: clustering=higher|lower|similar; path_length=shorter|longer|similar",
        "expect": {"clustering": "higher", "path_length": ["similar", "shorter"]},
    },
]

CASES += [
    {
        "id": "s3-kaliningrad-read",
        "why": "Live failure, 29 Sep: Pedro gave modern Kaliningrad's degrees as 2, 2, 3, 3 from memory; "
               "L01 p. 16 labels the four vertices 4, 3, 6 and 5 (only in the image, not the extracted text).",
        "user": 9106,
        "folder": "networkscience1-3",
        "section": 2,
        "section_title": "Königsberg",
        "message": "On the modern Kaliningrad slide (page 16), how many bridges does each of the four landmasses have, "
                   "and how many of them have an odd number of bridges?",
        "check": "CHECK: degrees=<the four numbers, comma-separated>; odd=<number>",
        "grade": "degrees",
        "expect": {"degrees": [3, 4, 5, 6], "odd": 2},
    },
    {
        "id": "s3-kaliningrad-student-right",
        "why": "Live failure, 29 Sep: the student correctly said an extra bridge keeps exactly two odd vertices; "
               "Pedro 'corrected' them with invented counts, marked the answer wrong and completed the section.",
        "user": 9106,
        "folder": "networkscience1-3",
        "section": 2,
        "section_title": "Königsberg",
        "until": 2560,
        "grade": "tutor_error",
        "facts": "L01 p. 16 labels the modern Kaliningrad vertices with degrees 4, 3, 6 and 5, so two are odd (3 and 5). "
                 "A new bridge between the bottom vertex (degree 5) and a top vertex (4 or 6) flips the parity of "
                 "exactly those two, so there are still exactly two odd vertices and the trail is still possible. The "
                 "student's answer is correct. Pedro's earlier claim that the islands have 3 bridges each and the banks "
                 "2 each was wrong.",
    },
]

# How Pedro teaches, judged on replay: the kind of question he asks and whether he has a voice.
# From the user's Knowledge Graphs lesson (29 Sep): mostly look-up questions, slide narration.
TEACHING_CASES = [
    {"id": "kg-opener", "until": 2592},
    {"id": "kg-after-edge", "until": 2598},
    {"id": "kg-partly-right", "until": 2602},
    {"id": "kg-after-count", "until": 2606},
]
for case in TEACHING_CASES:
    case.update(user=9106, folder="tetsing", section=2, section_title="Knowledge Graphs", grade="teaching")
CASES += TEACHING_CASES

# Checked on the captured context, not on a model reply.
ROUTING_CASES = [
    {
        "id": "global-routing",
        "why": "A graph theory question in global chat must pull Network Science material, not another course.",
        "message": "does my learner profile suggest i am already strong in basic graph theory",
        "expect_course": "Network Science",
    },
]
