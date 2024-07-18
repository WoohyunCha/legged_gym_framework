from urdf_generator import urdf_generator
import matplotlib.pyplot as plt

def main():
    infos = {} # dict of lists
    for i in range(100):
        u = urdf_generator()
        _, info = u.generate_urdf()
        for key, value in info.items():
            if not i:
                infos[key] = [value]
            else:
                infos[key].append(value)
    n = 0
    for key, _ in infos.items():
        print(key)
        plt.figure(n)
        plt.hist(infos[key], bins=10, alpha=0.75, color='blue', edgecolor='black')
        plt.title('Histogram of '+ key)
        plt.xlabel(key)
        plt.ylabel('Frequency')
        n += 1
    plt.show()
    
if __name__=="__main__":
    main()