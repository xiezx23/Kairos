
a_ttft = 1604; a_tpot = 36
b_ttft = 1050; b_tpot = 42
c_ttft = 1060; c_tpot = 36

for n in range(1, 100, 2):
    a_total = a_ttft + a_tpot*n
    b_total = b_ttft + b_tpot*n
    c_total = c_ttft + c_tpot*n
    print(f'speedup to W4A16: {a_total/c_total:4.2f}   to W8A8: {b_total/c_total:4.2f}')
