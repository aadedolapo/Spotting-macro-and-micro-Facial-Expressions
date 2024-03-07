from PIL import Image


def ConvertGraytoRGB(image):
    Image.open(image).convert("RGB").save(image)
    return "Converted successfully!"


if __name__ == '__main__':
    ConvertGraytoRGB()
