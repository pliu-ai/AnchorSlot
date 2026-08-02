# Historical reference result

These files preserve the completed July 30, 2026 inference artifact for pipeline
regression testing:

- 17 prediction cases;
- 225 present foreground label-cases;
- foreground Dice: 0.5167002009763103;
- foreground IoU: 0.4040599380946998;
- 31-class macro Dice: 0.47125917003260115.

The checkpoint was trained with Dataset202, which included the evaluation cases
in its training pool. These numbers must not be reported as held-out paper
performance.
