
import Image


def main():
	im = Image.new("RGB", (512, 512), "white")
	im.show()
	im.save("tmp.jpeg")
	


if __name__ == "__main__":
	main()
