# Define component lists with their specifications
cpus = [
	# Intel 14th Gen
	"Intel Core i9-14900KS",
	"Intel Core i9-14900KF",
	"Intel Core i9-14900K",
	"Intel Core i7-14700KF",
	"Intel Core i7-14700K",
	"Intel Core i5-14600KF",
	"Intel Core i5-14600K",
	# Intel 13th Gen
	"Intel Core i9-13900KS",
	"Intel Core i9-13900KF",
	"Intel Core i9-13900K",
	"Intel Core i7-13700KF",
	"Intel Core i7-13700K",
	"Intel Core i5-13600KF",
	"Intel Core i5-13600K",
	# AMD Ryzen 7000
	"AMD Ryzen 9 7950X3D",
	"AMD Ryzen 9 7950X",
	"AMD Ryzen 9 7900X3D",
	"AMD Ryzen 9 7900X",
	"AMD Ryzen 7 7800X3D",
	"AMD Ryzen 7 7700X",
	"AMD Ryzen 5 7600X"
]

gpus = [
	# NVIDIA RTX 40 Series
	"NVIDIA RTX 4090",
	"NVIDIA RTX 4080 Super",
	"NVIDIA RTX 4080",
	"NVIDIA RTX 4070 Ti Super",
	"NVIDIA RTX 4070 Ti",
	"NVIDIA RTX 4070 Super",
	"NVIDIA RTX 4070",
	"NVIDIA RTX 4060 Ti",
	"NVIDIA RTX 4060",
	# NVIDIA RTX 30 Series
	"NVIDIA RTX 3090 Ti",
	"NVIDIA RTX 3090",
	"NVIDIA RTX 3080 Ti",
	"NVIDIA RTX 3080",
	"NVIDIA RTX 3070 Ti",
	"NVIDIA RTX 3070",
	"NVIDIA RTX 3060 Ti",
	"NVIDIA RTX 3060",
	# AMD RX 7000
	"AMD RX 7900 XTX",
	"AMD RX 7900 XT",
	"AMD RX 7800 XT",
	"AMD RX 7700 XT",
	"AMD RX 7600"
]

rams = [
	# DDR5
	"DDR5 8000MHz",
	"DDR5 7600MHz",
	"DDR5 7200MHz",
	"DDR5 6800MHz",
	"DDR5 6400MHz",
	"DDR5 6000MHz",
	"DDR5 5600MHz",
	"DDR5 5200MHz",
	# DDR4
	"DDR4 4000MHz",
	"DDR4 3600MHz",
	"DDR4 3200MHz",
	"DDR4 3000MHz",
	"DDR4 2666MHz",
	"DDR4 2400MHz",
	"DDR4 2133MHz"
]


def get_combinations():
	# Generate all possible combinations of CPU, GPU and RAM
	combinations = []
	for cpu in cpus:
		for gpu in gpus:
			for ram in rams:
				combinations.append(f"CPU {cpu} with GPU {gpu} and RAM {ram}")
	print(combinations)
	return combinations


if __name__ == "__main__":
	get_combinations()