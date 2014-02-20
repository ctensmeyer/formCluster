
import kmeans


def get_data_set():
	data = [ (1,), (1,), (2,), (2,), (2,), (3,), (3,),
				(10,), (10,), (11,), (12,), (13,), (13,)]
	return data

def main():
	data = get_data_set()
	var = kmeans.KMeans(data, 2)
	var.cluster()
	var.display()

if __name__ == "__main__":
	main()
	
