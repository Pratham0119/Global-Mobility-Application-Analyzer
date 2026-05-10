import os


def get_latest_file(directory, keyword):
    """
    Returns latest file containing keyword in filename
    """
    files = [
        f for f in os.listdir(directory)
        if keyword in f
    ]

    if not files:
        return None

    files.sort(reverse=True)

    return os.path.join(directory, files[0])