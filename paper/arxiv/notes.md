
## title, references, and adresses
- PandaProbe is owned by Chirpz AI lab, so instead of "PandaProbe Research Lab" use "Chirpz AI lab"
- add my email as the corresponding author with address sina@pandaprobe.com

## abstract
- instead of LLM agents use AI agents
- in the abstract go with a more vision oriented story: "as frontier models are getting more capable, they're calling hundreds of tools, they're reading and writing from critical data sources, they're deployed in healthcare, military and finance; so as capability go up, complexity goes up, and the cost of issues goes up dramatically, and you have all these failure modes that you've never anticipated". with this you can layout the foundation on the problem that we're solving and our answer/solution for it.
- the abstract is overlly detailed in some parts, for example it talsk about "ts prompt carries one constant capability sentence and four read-only tools, and no rule text is ever injected", we can remove these stuff and simplify the technical depth in the abstract, focus on the vision, the problem, the solution we provide, and at the end one line of high level metrics and improvements we acheived.

## introduction
- following this update in the vision foundation in abstract, update the introduction to align with the updated core problem we're framign and solving.
- instead of "package-owned repair agent" or "packaged repair agent" use the term "repair engine" across the paper.
- instead of "developer’s task agent" simply use "harness" or "developer's harness". we don't want to use complex terms here. the repair agent in the loop is basically an agent works as the core repair engine of the loop so we could call it the "repair engine", the agent that developer builds and wraps it with pandaprobe harness is simply the "harness", in other words pandaprobe is providing the "harness building" developer tool for building harnesses that can learn from failures and self-repair.
- at the end of introduction section where it discusses what pandaprobe harness is we need to provide more clarity and details. PandaProbe core is an open source agent tracing and evaluation service. it's the foundation for tracing agents, and evaluate them in cloud. it comes with pandaprobe's sdk client for tracing agents built with any framework and it also comes with a CLI that provides a communication interface built for agents to interact with PandaProbe core service. then PandaProbe harness is a separate SDK that wraps the agent that developer is building and enables it with all the tooling for the self-repair loop.
- you must refine the contributions part with more clear values and contributions this work provides.

## background and related work
- rewrite, expand, and refine the backgroung work section. it's too shallow and useless right now. you don't need to necessarily add the references now, i can add them later just mark them as [placeholder] with red color for me find and repalce with actual references.


## PandaProbe Harness
overall the section 3 and its subsections look good, just see if you need to refine anything.
- one quick note: remove any references and mentions to AppWorld. for example i see it's being mention in section 3.3 The outcome verifier. check across the paper and remove these references and clean up.
- in section 3.4 instead of "Read-only rules from the harness workspace" use "rules from the harness workspace"

## experimental setup
- do not use the term "PandaBench" as it's a new benchmark we created, we're basically running tau2 and terminal bench just testing them original vs wrapped with PandaProbe. so i don't know if it's correct to use the term PandaBench or not, if it's ok just to use a wording that does not imply we're creating a new benchmark.
- update and refine the intro to this section. use an alternative term for "task models", also use alternative term for "aseline–Harness comparison".
- following previous instructions update the terms such as "task-agen" to follow the new pattern across the paper.
- in section "4.3 Continuous operation" instead of "Harness remains active" use a term like "repair loop remains active" or "repair engine remains active"

## results and discussion
- update the section 5.1 Main results and add analysis after the table.
- table 6 Forward-trial validation path is missing some rows, fill these rows based on the results in paper/arxiv/paper_tables.md
- table 7 Within-run learning dynamics  is missing some rows, fill these rows based on the results in paper/arxiv/paper_tables.md
- table 8 Rule lifecycle by recorded is missing some rows, fill these rows based on the results in paper/arxiv/paper_tables.md
- table 9 Harness activity by recorded run is missing some rows, fill these rows based on the results in paper/arxiv/paper_tables.md
- table 10 Cost and overhead by matched run is missing some rows, fill these rows based on the results in paper/arxiv/paper_tables.md
- refine the limitations, but keep them crisp and to the point.

## conclusion
- remove the future work part, we need to focus on conclusion and we do not want to promise anything
- rewrite and refine the conclusion

