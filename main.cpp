#include "ndarray.h"

using namespace std;
int main()
{
    NDarray<int> arr1D({5}, 7);
    cout << "Tableau 1D: " << endl;
    std::cout << "Il y a: " << arr1D.getSize() << " éléments" << endl;
    arr1D.at({2}) = 9;
    arr1D.print();

    NDarray<int> arr1 = NDarray<int>::arange(0, 10, 2); // Tableau avec valeurs allant de 0 à 10 avec un pas de 2
    std::cout << "arange(0, 10, 2):" << std::endl;
    arr1.print(); // Affiche le tableau

    // Exemple d'utilisation de linspace
    NDarray<int> arr2 = NDarray<int>::linspace(1, 10, 4); // Tableau avec 5 valeurs espacées uniformément entre 0 et 1
    std::cout << "linspace(1, 10, 4):" << std::endl;
    arr2.print(); // Affiche le tableau

    NDarray<int> arr2D = NDarray<int>::full({3, 4}, 5);
    cout << "\nTableau 2D: " << endl;
    std::cout << "Il y a: " << arr2D.getSize() << " éléments" << endl;
    std::vector<size_t> shape = arr2D.getShape();
    std::cout << "Tableau de dimensions: { ";
    for (size_t dim : shape)
    {
        std::cout << dim << " ";
    }
    std::cout << "}";
    cout << endl;
    arr2D.at({1, 2}) = 9; // Remplacé la 1ère  ligne et deuxième colonne du matrice par 9
    arr2D.print();
    // Redimensionner en 3x2
    arr2D.reshape({4, 3});
    std::cout << "\nTableau après reshape (4x3) :" << std::endl;
    arr2D.print();
    NDarray<int> flattened = arr2D.flatten();
    std::cout << "\nTableau aplati en 1D :" << std::endl;
    flattened.print();

    NDarray<int> arr3D = NDarray<int>::ones({2, 3, 4});
    cout << "\nTableau 3D: " << endl;
    arr3D.print();

    NDarray<int> identity = NDarray<int>::eye(4);
    cout << "\nMatrice Identité 4x4: " << endl;
    identity.print();

    return 0;
}