import re

def parse_amber_frcmod(file_path):
    data = {
        "MASS": {},
        "BOND": {},
        "ANGLE": {},
        "DIHE": {},
        "NONBON": {}
    }

    section = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line in data:
                section = line
                continue

            parts = re.split(r"\s+", line)

            if section == "MASS" and len(parts) >= 2:
                data["MASS"][parts[0]] = float(parts[1])

            elif section == "BOND" and len(parts) >= 3:
                data["BOND"][parts[0]] = (float(parts[1]), float(parts[2]))

            elif section == "ANGLE" and len(parts) >= 3:
                data["ANGLE"][parts[0]] = (float(parts[1]), float(parts[2]))

            elif section == "DIHE" and len(parts) >= 4:
                data["DIHE"][parts[0]] = tuple(map(float, parts[1:4]))

            elif section == "NONBON" and len(parts) >= 3:
                data["NONBON"][parts[0]] = (float(parts[1]), float(parts[2]))

    return data
