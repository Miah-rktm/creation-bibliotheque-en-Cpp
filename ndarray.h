#ifndef NDARRAY_H
#define NDARRAY_H
#include <iostream>
#include <vector>
#include <string>
#include <initializer_list>

template <typename T>
class NDarray
{
public:
    NDarray(std::initializer_list<size_t> dims, T value = T());
    void print() const;
    size_t getSize() const;                                              // fonction retournant le nombre d'éléments dans le tableau n-dimensionnel
    std::vector<size_t> getShape() const { return shape; };              // fonction retournant le dimension du tableau n-dimensionnel
    static NDarray<T> zeros(std::initializer_list<size_t> dims);         // fonction permettant de créer un tableau rempli de 0
    static NDarray<T> ones(std::initializer_list<size_t> dims);          // rempli de uns
    static NDarray<T> full(std::initializer_list<size_t> dims, T value); // rempli de valeurs
    static NDarray<T> eye(size_t n);                                     // matrice idnetité 4*4
    static NDarray<T> arange(T start, T stop, T step);                   // fonction :creer un tableau avec un pas défini
    static NDarray<T> linspace(T start, T stop, size_t num);             // fonction :creer un tableau de valeurs espacées uniformément
    T &at(std::initializer_list<size_t> indices);                        // accéder aux éléments du tableau
    void reshape(std::initializer_list<size_t> new_shape);               // changer la forme du tableau
    NDarray<T> flatten() const;                                          // aplatir le tableau en 1D
    NDarray<T> concatenate(const NDarray<T> &other, size_t axis = 0);    //Concatenation sur une dimension
    NDarray<T> hstack(const NDarray<T> &other);                          //Concatenation horizontale
    NDarray<T> vstack(const NDarray<T> &other);                          //Concatenation verticale
    static NDarray<double> rand(std::initializer_list<size_t> dims);     // Nombres aléatoires entre 0 et 1

private:
    std::vector<T> tab;        // tab sous forme linéaire
    std::vector<size_t> shape; // dimensions du tableau sous forme de vecteurs
};
#include "ndarray.tpp"
#endif
