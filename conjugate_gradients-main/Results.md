# EUMaster4HPC-Challenge - Results

## Sequential 
| Matrix Size | Avg Time  | Number of Samples |
|-------------|-----------|-------------------|
| 30000       | 197.023 s | 3                 |

## OpenMP
| Number of Threads | Matrix Size | Avg Time  | Number of Samples |
|-------------------|-------------|-----------|-------------------|
| 1                 | 30000       | 327.362 s | 3                 |
| 4                 | 30000       | 89.557 s  | 3                 |
| 8                 | 30000       | 81.720 s  | 3                 |
| 12                | 30000       | 82.032 s  | 3                 |
| 16                | 30000       | 82.136 s  | 3                 |
| 24                | 30000       | 74.728 s  | 3                 |
| 32                | 30000       | 70.728 s  | 3                 |
| 64                | 30000       | 77.931 s  | 3                 |

## OpenACC


## MPI + OpenMP