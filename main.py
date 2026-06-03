import pdftxt_converter
import os
if __name__ == '__main__':
    folder_path = "refined_data"

    # check if folder is empty
    if not os.listdir(folder_path):
        print("Folder is empty → calling data extractor")
        pdftxt_converter.ingestion_pipeline()  # call your function here
    else:
        print("source data ready ")
