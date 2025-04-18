#include "ndarray.h"
#include <iostream>

using namespace std;

// Fonction affichant la forme:dimensions du tableau
void printShape(const vector<size_t> &shape)
{
    cout << "Shape: { ";
    for (size_t dim : shape)
    {
        cout << dim << " ";
    }
    cout << "}" << endl;
}

int main()
{
    // Test du constructeur avec initialisation
    NDarray<int> arr1({2, 3}, 5);
    cout << "Tableau 2x3 initialisé à 5 : " << endl;
    arr1.print();
    printShape(arr1.getShape());

    // Test du constructeur avec vector
    NDarray<int> arr2({4, 3}, 10);
    cout << "\nTableau 4x3 initialisé à 10 : " << endl;
    arr2.print();
    printShape(arr2.getShape());

    // Test de la méthode full
    NDarray<int> arrFull = NDarray<int>::full({2, 2}, 7);
    cout << "\nTableau full(2, 2, 7) : " << endl;
    arrFull.print();

    // Test de la méthode zeros
    NDarray<int> arrZeros = NDarray<int>::zeros({3, 3});
    cout << "\nTableau zeros(3, 3) : " << endl;
    arrZeros.print();

    // Test de la méthode ones
    NDarray<int> arrOnes = NDarray<int>::ones({3, 3});
    cout << "\nTableau ones(3, 3) : " << endl;
    arrOnes.print();

    // Test de la méthode arange
    NDarray<int> arrRange = NDarray<int>::arange(0, 10, 2);
    cout << "\nTableau arange(0, 10, 2) : " << endl;
    arrRange.print();

    // Test de la méthode linspace
    NDarray<double> arrLinspace = NDarray<double>::linspace(1, 10, 5);
    cout << "\nTableau linspace(1, 10, 5) : " << endl;
    arrLinspace.print();

    // Test de la méthode rand
    NDarray<int> arrRand = NDarray<int>::rand({3, 3}, 0, 100);
    cout << "\nTableau rand(3, 3, 0, 100) : " << endl;
    arrRand.print();

    // Test de la méthode randint
    NDarray<int> arrRandInt = NDarray<int>::randint(0, 10, {2, 3});
    cout << "\nTableau randint(0, 10, (2, 3)) : " << endl;
    arrRandInt.print();

    // Test de l'indexation
    cout << "\nAccès à arr1[0, 1] : " << arr1.at({0, 1}) << endl;

    // Test du slicing
    NDarray<int> arrSliced = arr2[{Slice(1, 3), Slice(0, 2)}];
    cout << "\nSlicing arr2[1:3, 0:2] : " << endl;
    arrSliced.print();

    // Test de slicing avec indices négatifs
    NDarray<int> arr1D = NDarray<int>::arange(0, 10, 1);
    cout << "\nSlicing avec indices négatifs arr1D[-3:-1] : " << endl;
    NDarray<int> sliceNeg = arr1D[{Slice(-3, -1)}];
    sliceNeg.print();

    // Test de slicing avec pas négatif
    cout << "\nSlicing avec pas négatif arr1D[::-1] : " << endl;
    NDarray<int> sliceRev = arr1D[{Slice(-1, -11, -1)}];
    sliceRev.print();

    NDarray<int> arr2D({4, 4}, 0);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            arr2D.at({i, j}) = i * 4 + j;
    cout << "\nSlicing arr2D[1:, :2] : " << endl;
    NDarray<int> subMatrix = arr2D[{Slice(1, 4), Slice(0, 2)}];
    subMatrix.print(); // Devrait afficher [[4, 5], [8, 9], [12, 13]]

    NDarray<int> arr({3, 3}, 0);
    for (size_t i = 0; i < 3; ++i)
        arr.at({i, 1}) = 10 + i;
    cout << "\nSlicing arr2D[:, 1] : " << endl;
    NDarray<int> col1 = arr[{Slice(0, 3), Slice(1, 2)}];
    col1.print(); // Devrait afficher [[10], [11], [12]] ou [10, 11, 12] selon l'affichage

    // Test de reshape
    NDarray<int> arrReshaped = arr2;
    arrReshaped.reshape({6, 2});
    cout << "\nReshape de arr2 de (4, 3) à (6, 2) : " << endl;
    arrReshaped.print();

    // Test de flatten
    NDarray<int> arrFlattened = arr2.flatten();
    cout << "\nFlatten de arr2 : " << endl;
    arrFlattened.print();

    // Test de la concaténation horizontale (hstack)
    NDarray<int> arr2_mod = NDarray<int>::full({3, 3}, 10);
    NDarray<int> arr1_mod = NDarray<int>::full({3, 3}, 5);
    NDarray<int> arrConcat = arr2_mod.hstack(arr1_mod);
    cout << "\nConcaténation horizontale arr2_mod et arr1_mod : " << endl;
    arrConcat.print();

    // Test de la concaténation verticale (vstack)
    NDarray<int> arrVConcat = arr2.vstack(arr1);
    cout << "\nConcaténation verticale arr2 et arr1 : " << endl;
    arrVConcat.print();

    // Opérations arithmétiques
    NDarray<int> arrA({2, 2}, 4);
    NDarray<int> arrB({2, 2}, 2);

    cout << "\nAddition (arrA + arrB) : " << endl;
    NDarray<int> sumArr = NDarray<int>::add(arrA, arrB);
    sumArr.print();

    cout << "\nSoustraction (arrA - arrB) : " << endl;
    NDarray<int> subArr = NDarray<int>::subtract(arrA, arrB);
    subArr.print();

    cout << "\nMultiplication (arrA * arrB) : " << endl;
    NDarray<int> mulArr = NDarray<int>::multiply(arrA, arrB);
    mulArr.print();

    cout << "\nDivision (arrA / arrB) : " << endl;
    NDarray<int> divArr = NDarray<int>::divide(arrA, arrB);
    divArr.print();

    // Opérations avec scalaires
    cout << "\nAddition avec scalaire (arrA + 5) : " << endl;
    NDarray<int> scalarAdd = arrA + 5;
    scalarAdd.print();

    cout << "\nMultiplication avec scalaire (arrA * 2) : " << endl;
    NDarray<int> scalarMul = arrA * 2;
    scalarMul.print();

    // Produit matriciel
    NDarray<int> mat1({2, 3}, 1);
    NDarray<int> mat2({3, 2}, 2);
    NDarray<int> dotResult = NDarray<int>::dot(mat1, mat2);
    cout << "\nProduit matriciel dot(mat1, mat2) : " << endl;
    dotResult.print();

    // Test pour un tableau 3D
    NDarray<int> arr3D({2, 3, 4}, 1);
    cout << "\nTableau 3D initialisé à 1 (shape {2, 3, 4}) : " << endl;
    arr3D.print();

    // Test de slicing 3D
    NDarray<int> arr3DSliced = arr3D[{Slice(0, 2), Slice(1, 3), Slice(0, 4, 2)}];
    cout << "\nSlicing 3D arr3D[0:2, 1:3, 0:4:2] : " << endl;
    arr3DSliced.print();

    // Test de randint 4D
    NDarray<int> arr4DRandInt = NDarray<int>::randint(0, 10, {2, 2, 2, 2});
    cout << "\nTableau 4D randint(0, 10, (2, 2, 2, 2)) : " << endl;
    arr4DRandInt.print();

    return 0;
}
