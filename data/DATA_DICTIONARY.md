# Data dictionary – `data/raw/DataP.csv`

Source: laboratory experiments of Ghorbany, Hamedi & Ghodsian (2023), *Urban Water Journal* 20(1). 135 rows = 9 soil beds × 5 pressure heads × 3 observation times. The raw file is never modified by the code.

| Column | Unit | Meaning |
|---|---|---|
| D50 | mm | median grain size of the sand (0.64, 1.26, **2.1**) |
| CD | mm | cover depth (pipe crown to ground surface): 400, 500, 600 |
| P | m H2O | pipe pressure head: 15, 25, 35, 45, 55 |
| t | min | time of the image: 10, 20, 30 |
| T | min | reference duration (30) |
| Af | **cm²** | projected (front view) fluidized-zone area from digitised images |

Derived in `fluidml/data.py`: `bed` (= soil × cover depth; the grouping unit for all validation), `X = P/sqrt(CD·d50)` (all lengths in m; dimensionless), `P_CD`, `CD_d50`, `tau = t/T`, `Z = Af[mm²]/d50[mm]²`, `Y = log10(Z)`.

Known data issue: the record D50=0.64, CD=600, P=15, t=10 (Af = 32.28) breaks the expected monotonic growth with time (4.50 and 5.39 at 20 and 30 min); see `results/tables/audit_nonmonotonic_records.csv`.
