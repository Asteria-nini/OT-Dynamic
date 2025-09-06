import os
import xml.etree.ElementTree as ET
import pandas as pd


def parse_xml(file_path):
    # Parse XML file and return dictionary representation
    tree = ET.parse(file_path)
    root = tree.getroot()

    def parse_element(element):
        # Recursively parse XML elements
        parsed = {}
        for child in element:
            if len(child) == 0:
                parsed[child.tag] = child.text
            else:
                parsed[child.tag] = parse_element(child)
        return parsed

    return parse_element(root)


def convert_to_dataframe(parsed_data):
    # Convert parsed data to Pandas DataFrame
    if isinstance(parsed_data, list):
        return pd.DataFrame(parsed_data)
    else:
        return pd.DataFrame([parsed_data])


def save_to_csv(df, file_name, output_dir):
    # Save DataFrame as CSV file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False, encoding='utf-8')


def batch_process_xml_in_directory(input_dir, output_dir):
    # Process all XML files in directory and save as CSV
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing: {file_path}")

            parsed_data = parse_xml(file_path)
            df = convert_to_dataframe(parsed_data)
            csv_file_name = filename.replace(".xml", ".csv")
            save_to_csv(df, csv_file_name, output_dir)

            print(f"Saved: {os.path.join(output_dir, csv_file_name)}")


if __name__ == "__main__":
    base_input_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\raw"
    base_output_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\interim"

    # Process folders G_1 to G_8
    for i in range(1, 9):
        folder_name = f"G_{i}"
        input_directory = os.path.join(base_input_dir, folder_name)
        output_directory = os.path.join(base_output_dir, folder_name)

        if os.path.exists(input_directory):
            print(f"\nProcessing folder: {input_directory}")
            batch_process_xml_in_directory(input_directory, output_directory)
        else:
            print(f"\nFolder not found: {input_directory}")