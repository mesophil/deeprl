import os, glob

def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    imgPath = os.path.join(currentDir, "../globulus")

    # for f in os.listdir(imgPath):
    #     if not f.endswith('.png'): continue
    #     os.remove(os.path.join(imgPath, f))

    pathsList = glob.glob(imgPath + '/**/*', recursive=True)

    for p in pathsList:
        if os.path.isfile(p) and p.endswith('.png'):
            os.remove(p)
            print(f"Removed: {p}")

    for p in pathsList:
        if os.path.isdir(p):
            os.rmdir(p)

if __name__ == "__main__":
    main()