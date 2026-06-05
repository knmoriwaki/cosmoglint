Omega_m = 0.3089
sigma_8 = 0.8159

A_SN1 = 1.0
A_AGN1 = 1.0
A_SN2 = 1.0
A_AGN2 = 1.0

print("#Name Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2 num")

run_name = "Test"
count = 0
for A_SN1 in [0.8, 1, 1.2]:
    print(f"{run_name}_{count} {Omega_m:.4f} {sigma_8:.4f} {A_SN1:.4f} {A_AGN1:.4f} {A_SN2:.4f} {A_AGN2:.4f} {count}")

    count += 1