# fetch_weights.py
import modal

from modal_train import list_weights, get_weight_file

if __name__ == "__main__":
    # List files available in the Modal volume
    list_weights.remote()

    # Download 'best.pt' (or any file)
    data = get_weight_file.remote("best.pt")
    with open("weights/best.pt", "wb") as f:
        f.write(data)
    print("Downloaded weights/best.pt")
