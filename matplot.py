import numpy as np
from matplotlib import pyplot as plt
from skimage import measure


# Question 1
def generate_random_numbers():
    random_numbers = np.random.randn(100)
    return random_numbers


# Question 2
def plot_random_numbers():
    random_numbers = generate_random_numbers()
    plt.plot(random_numbers)
    plt.show()


# Question 3
def multiple_plots():
    x = np.linspace(0, 2, 100)
    plt.plot(x, x, label='linear', color='blue', linestyle='dashed')
    plt.plot(x, x ** 2, label='quadratic', color='red', linestyle='dotted')
    plt.plot(x, x ** 3, label='cubic', color='green', linestyle='dashdot')
    plt.plot(x, np.exp(x), label='exponential', color='purple', linestyle='solid')
    plt.axis([0, 2, 0, 10])
    plt.xlabel('x-axis', fontsize=10, color='blue')
    plt.ylabel('y-axis', fontsize=10, color='blue')
    plt.title("Multiple plots", fontsize=15, color='red')
    plt.legend()
    plt.show()


# Question 4
def plot_with_arrows():
    random_numbers = generate_random_numbers()

    plt.plot(random_numbers, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    plt.xlabel('Absciss', fontsize=16, color='blue')
    plt.ylabel('Ordinate', fontsize=16, color='blue')

    plt.legend(['Data points'], facecolor='yellow', bbox_to_anchor=(0.2, 1.1), loc='upper right')

    max_value = np.max(random_numbers)
    max_index = np.argmax(random_numbers)
    plt.annotate('Max value', xy=(max_index, max_value), xytext=(max_index + 10, max_value + 10),
                 arrowprops=dict(facecolor='black', shrink=0.01))
    plt.plot(max_index, max_value, 'ro', markersize=12)
    plt.text(max_index + 3, max_value, 'Max value', fontsize=10, color='green')

    min_value = np.min(random_numbers)
    min_index = np.argmin(random_numbers)
    plt.annotate('Min value', xy=(min_index, min_value), xytext=(min_index + 10, min_value - 10),
                 arrowprops=dict(facecolor='black', shrink=0.01))
    plt.plot(min_index, min_value, 'ro', markersize=12)
    plt.text(min_index + 3, min_value, 'Min value', fontsize=10, color='green')

    plt.show()


# Question 5.a
def plot_histogram():
    random_numbers = generate_random_numbers()
    plt.hist(random_numbers, bins=20, rwidth=0.8)
    plt.xlabel('X axis', fontsize=16, color='blue')
    plt.ylabel('Y axis', fontsize=16, color='blue')
    plt.legend(['Random numbers'], facecolor='yellow', bbox_to_anchor=(0.2, 1.1), loc='upper right')
    plt.show()


# Question 5.b
def plot_pie():
    slices = [7, 2, 2, 13]
    languages = ['JavaScript', 'C++', 'Java', 'Python']
    cols = ['c', 'm', 'r', 'b']
    plt.pie(slices, labels=languages, colors=cols, startangle=90, shadow=True, explode=(0, 0.1, 0, 0),
            autopct='%1.1f%%')
    plt.title('Prefered programming language')
    plt.show()


# Question 6
def plot_2d_into_3d_space():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Make data.
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.exp(2 * np.cos(R))
    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap='gist_earth', edgecolor='none', linewidth=0.1,
                    antialiased=True, alpha=0.6, shade=True, rstride=1, cstride=1)
    plt.show()


def plot_heart():
    # Set up mesh
    n = 100

    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    z = np.linspace(-3, 3, n)
    X, Y, Z = np.meshgrid(x, y, z)

    # Create cardioid function
    def f_heart(x, y, z):
        F = 320 * ((-x ** 2 * z ** 3 - 9 * y ** 2 * z ** 3 / 80) +
                   (x ** 2 + 9 * y ** 2 / 4 + z ** 2 - 1) ** 3)
        return F

    # Obtain value to at every point in mesh
    vol = f_heart(X, Y, Z)

    # Extract a 2D surface mesh from a 3D volume (F=0)
    verts, faces, normals, values = measure.marching_cubes(vol, 0, spacing=(0.1, 0.1, 0.1))

    # Create a 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap='spring', lw=1, alpha=0.5)

    # Change the angle of view and title
    ax.view_init(15, -15)

    ax.set_title("Heart Plot", fontsize=15)

    plt.show()


if __name__ == "__main__":
    # Question 1
    # print(generate_random_numbers())

    # Question 2
    # plot_random_numbers()

    # Question 3
    # multiple_plots()

    # Question 4
    # plot_with_arrows()

    # Question 5
    # plot_histogram()
    # plot_pie()

    # Question 6
    # plot_2d_into_3d_space()
    # Question 6 : Heart
    plot_heart()
