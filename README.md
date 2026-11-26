# MATH 463 Mathematical Biology: lecture notes and homework

Course materials for MATH 463 (Mathematical Biology), a 400-level course in the Department of Mathematics at the University of Michigan, taught by Naoki Masuda in Fall 2026. The course covers how differential equations and other mathematical and computational methods are used to model biological systems, from populations and epidemics to cells, molecules, networks, and evolution.

Anyone is welcome to read, use, and adapt these materials.

## What is shared here, and what is not (yet)

I share only the materials that I can legally share:

- **Materials I created from scratch.** Some of my lecture notes build on notes from a previous instructor of this course, either directly or in modified form. I do not post those notes.
- **Copyright-cleared material only.** In class, my lecture notes show many figures taken from published papers and books. I cannot post those notes publicly until the figures are cleared, redrawn, or replaced with links.

So this repository is incomplete for now. My goal is to share everything eventually, and I will add materials as they become ready.

## Course structure

| Section | Topic | Shared |
|---|---|---|
| 1 | Introduction | not yet |
| 2 | One-dimensional ODE models | not yet |
| 3 | Higher-dimensional ODE models, phase plane, linear stability analysis, and numerical solutions | not yet |
| 4 | Epidemic dynamics | not yet |
| 5 | Oscillation and synchronization | not yet |
| 6 | Data-driven discovery of dynamical systems | [lecture notes](lecture-notes/) |
| 7 | Chemical kinetics | not yet |
| 8 | AI for mathematical biology | [lecture notes](lecture-notes/) |
| 9 | Network biology and network medicine | not yet |
| 10 | Mathematical ecology | not yet |
| 11 | Evolutionary game theory | not yet |

| Homework | Covers sections | Shared |
|---|---|---|
| HW1 | 2 | not yet |
| HW2 | 3 | not yet |
| HW3 | 4 and 5 | not yet |
| HW4 | 6 and 7 | [yes](homework/) |
| HW5 | 9, 10, and 11 | [yes](homework/) |

Several sections, and HW1–HW3, draw on S. H. Strogatz, *Nonlinear Dynamics and Chaos*, 2nd edition (CRC Press, 2015). HW1–HW3 are not released for now. I will also post the syllabus later.

## Designed for the AI era

These materials are meant to be used together with AI tools, and that is the point of sharing them in this form.

- **LaTeX source is included.** Each document comes as both `.tex` and `.pdf`. An AI assistant reads the `.tex` far more accurately than the PDF, so you can give it the source and ask questions, request more examples, or check your own reasoning.
- **Homework comes with full solutions.** In the age of AI, it no longer makes sense to ask students to solve problems and then simply mark their answers right or wrong; the answers are easy to obtain, and the grading teaches little. Instead, each homework set is a self-study resource: all problems first, then detailed solutions, to be worked through at the student's own pace with or without AI help. I assess mastery of the mathematics in in-person midterm exams.
- **Section 8** of the lecture notes explains my thinking on how students can use AI to learn mathematics and to do research projects.

## AI disclosure

I used AI tools, mainly Claude (Anthropic) and ChatGPT (OpenAI, through the University of Michigan's service), to prepare these materials. The homework sets were generated with AI from my lecture notes. I checked, corrected, and revised everything, and I take full responsibility for the content. If you find an error, please open an issue.

## About the format of the lecture notes

The lecture notes are not formatted like a textbook. They use a large font and terse bullet points because I show them directly on the classroom screen and scroll through them gradually in front of the students. I share them as they are, rather than polishing them into a different form.

## Repository layout

```
lecture-notes/                    lecture notes (.tex and .pdf)
lecture-notes/figures/<section>/  figures used in each section of the notes
homework/                         homework sets with solutions (.tex and .pdf)
homework/figures/homework/        figures used in the homework, with the Python scripts that generate them
```

Figures taken from published papers are used under their Creative Commons licenses (CC BY 4.0 or CC BY 3.0), and each caption names the source, links its DOI, and states its license.

To rebuild a PDF, run `pdflatex` on the `.tex` file from within its folder.

## License

Text and my own figures: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt them for any purpose, provided you give appropriate credit. Figures from published papers remain under the license stated in their captions.

Code (the Python scripts): [MIT License](https://opensource.org/license/mit).

## Contact

[Naoki Masuda](https://naokimas.github.io/naokimasuda.github.io/)
