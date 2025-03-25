#include "ndarray.h"
#include <iostream>

using namespace std;

void printShape(const vector<size_t> &shape)
{
    cout << "Dimensions: { ";
    for (size_t dim : shape)
    {
        cout << dim << " ";
    }
    cout << "}" << endl;
}

int main()
{
    cout << "=== Tests des constructeurs et méthodes de base ===" << endl;
    NDarray<int> arr1D({5}, 7);
    cout << "\nTableau 1D initialise avec 7:" << endl;
    cout << "Nombre d'elements: " << arr1D.getSize() << endl;
    arr1D.at({2}) = 9;
    arr1D.print();

    NDarray<int> arr2D = NDarray<int>::full({3, 4}, 5);
    cout << "\nTableau 2D initialise avec 5:" << endl;
    cout << "Nombre d'elements: " << arr2D.getSize() << endl;
    printShape(arr2D.getShape());
    arr2D.at({1, 2}) = 9;
    arr2D.print();

    NDarray<int> arr3D = NDarray<int>::ones({2, 3, 4});
    cout << "\nTableau 3D initialise avec 1:" << endl;
    arr3D.print();

    cout << "\n=== Tests des generateurs de tableaux ===" << endl;
    NDarray<int> arrRange = NDarray<int>::arange(0, 10, 2);
    cout << "Tableau avec arange(0, 10, 2):" << endl;
    arrRange.print();

    NDarray<int> arrLinspace = NDarray<int>::linspace(1, 10, 4);
    cout << "\nTableau avec linspace(1, 10, 4):" << endl;
    arrLinspace.print();

    NDarray<int> identity = NDarray<int>::eye(4);
    cout << "\nMatrice identite 4x4:" << endl;
    identity.print();

    cout << "\n=== Tests de manipulation de forme ===" << endl;
    cout << "Tableau 2D avant reshape:" << endl;
    arr2D.print();
    arr2D.reshape({4, 3});
    cout << "\nTableau 2D apres reshape en 4x3:" << endl;
    arr2D.print();

    NDarray<int> flattened = arr2D.flatten();
    cout << "\nTableau 2D aplati en 1D:" << endl;
    flattened.print();

    cout << "\n=== Tests des operations de concatenation ===" << endl;
    NDarray<int> concatResult = arrRange.concatenate(arrLinspace);
    cout << "Concatenation de arrRange et arrLinspace (axis=0):" << endl;
    concatResult.print();

    NDarray<int> arr2d_1 = NDarray<int>::full({2, 3}, 1);
    NDarray<int> arr2d_2 = NDarray<int>::full({2, 2}, 2);
    cout << "\nTableau 2D_1:" << endl;
    arr2d_1.print();
    cout << "Tableau 2D_2:" << endl;
    arr2d_2.print();
    NDarray<int> hstackResult = arr2d_1.hstack(arr2d_2);
    cout << "Hstack (concatenation horizontale):" << endl;
    hstackResult.print();

    NDarray<int> arr2d_3 = NDarray<int>::full({1, 3}, 3);
    cout << "\nTableau 2D_1:" << endl;
    arr2d_1.print();
    cout << "Tableau 2D_3:" << endl;
    arr2d_3.print();
    NDarray<int> vstackResult = arr2d_1.vstack(arr2d_3);
    cout << "Vstack (concatenation verticale):" << endl;
    vstackResult.print();

    NDarray<double> randArray = NDarray<double>::rand({3, 3});
    cout << "Tableau 3x3 de nombres aleatoires (0-1):" << endl;
    randArray.print();

    return 0;
}
