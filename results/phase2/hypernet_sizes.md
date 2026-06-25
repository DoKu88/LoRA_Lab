# T3 (size half) — hypernet params per output parameterization

Mistral-7B target set: q/k/v x 32 layers = 96 targets, rank 16, d_task 384. **Committed default: VeRA** (measured generator: 55.84 M params).

| parameterization | hypernet params (M) |
|---|--:|
| vera | 55.68 |
| lowrank | 151.43 |
| full | 2651.85 |
