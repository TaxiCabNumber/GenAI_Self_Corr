import re

def extract_number_after_pounds(response):
    """
    Extracts the number after 'I hope it is correct' or '####' from the response string.

    Args:
        response (str): The response string.

    Returns:
        str: The extracted number or an empty string if no match is found.
    """
    last_correct_index = response.rfind('I hope it is correct')
    last_pounds_index = response.rfind('####')

    if last_correct_index != -1 and (last_correct_index > last_pounds_index or last_pounds_index == -1):
        split_part = response[last_correct_index + len('I hope it is correct'):]
    elif last_pounds_index != -1:
        split_part = response[last_pounds_index + len('####'):]
    else:
        return ""

    # Extract numbers from the split part
    match = re.search(r'(\$?\d+)', split_part)
    if match:
        return match.group(1)
    return ""

parse = "1. Clips sold in April: 48\n2. Clips sold in May: 48 / 2 = 24\n3. Total clips sold: 48 + 24 = 72\n\nI hope it is correct: 72 $clips$\n"
ground_truth = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"

print(extract_number_after_pounds(parse)) # 72
# print(extract_number_after_pounds(ground_truth)) # 72