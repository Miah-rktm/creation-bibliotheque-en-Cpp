#ifndef NDARRAY_H
#define NDARRAY_H

#include <iostream>
#include <vector>
#include <string>
#include <initializer_list>
#include <random>
#include <stdexcept>

// Déclaration anticipée de la classe NDarray pour utilisation dans la structure Slice
template <typename T>
class NDarray;

// Structure pour représenter une tranche (slice)
// Une tranche définit une sous-partie d'un tableau avec un début, une fin et un pas
struct Slice
{
    long long start, stop, step; // Utilisation de long long pour gérer les indices négatifs
    Slice(long long s, long long e, long long st = 1) : start(s), stop(e), step(st) {}
    Slice(long long idx) : start(idx), stop(idx + 1), step(1) {}
};

// Classe NDarray : implémente un tableau N-dimensionnel inspiré de NumPy
template <typename T>
class NDarray
{
public:
    // Initialise un tableau avec des dimensions spécifiées par une liste d'initialisation
    NDarray(std::initializer_list<size_t> dims, T value = T());
    // Initialise un tableau avec des dimensions spécifiées par un vecteur
    NDarray(std::vector<size_t> dims, T value = T());

    // Affichage
    // Affiche le tableau dans un format lisible, similaire à NumPy
    void print() const;

    // Retourne le nombre total d'éléments dans le tableau
    size_t getSize() const;
    std::vector<size_t> getShape() const { return shape; } // Retourne la forme (dimensions) du tableau

    static NDarray<T> zeros(std::initializer_list<size_t> dims);
    static NDarray<T> ones(std::initializer_list<size_t> dims);
    static NDarray<T> full(std::initializer_list<size_t> dims, T value);
    static NDarray<T> eye(size_t n);
    static NDarray<T> arange(T start, T stop, T step);                                // tableau 1D avec une séquence linéaire de valeurs
    static NDarray<T> linspace(T start, T stop, size_t num);                          // tableau 1D avec des valeurs équidistantes
    static NDarray<T> rand(std::initializer_list<size_t> dims, T min_val, T max_val); // tableau avec des valeurs aléatoires dans une plage donnée
    /**
     * Génère un tableau N-dimensionnel d'entiers aléatoires dans l'intervalle [low, high).
     * @param low Borne inférieure (incluse).
     * @param high Borne supérieure (exclue).
     * @param dims Forme du tableau résultant.
     * @return NDarray<T> avec des entiers aléatoires.
     */
    static NDarray<T> randint(T low, T high, std::initializer_list<size_t> dims);

    // Indexation
    // Accède à un élément spécifique via une liste d'indices (version modifiable)
    T &at(std::initializer_list<size_t> indices);
    // Accède à un élément spécifique via une liste d'indices (version constante)
    const T &at(std::initializer_list<size_t> indices) const;
    // Extrait une sous-partie du tableau via une liste de tranches (slicing)
    NDarray<T> operator[](const std::vector<Slice> &slices) const;

    // Manipulation de forme
    /**
     * Reformate le tableau à une nouvelle forme tout en préservant les données.
     * @param new_shape Nouvelle forme du tableau.
     * @throws std::runtime_error Si la nouvelle forme est incompatible avec la taille actuelle.
     */
    void reshape(std::initializer_list<size_t> new_shape);
    // Aplatit le tableau en un tableau 1D
    NDarray<T> flatten() const;
    // Concatène deux tableaux le long d'un axe spécifié
    NDarray<T> concatenate(const NDarray<T> &other, size_t axis);
    // Concatène deux tableaux horizontalement
    NDarray<T> hstack(const NDarray<T> &other);
    // Concatène deux tableaux verticalement
    NDarray<T> vstack(const NDarray<T> &other);

    // Opérateurs arithmétiques élément par élément
    // Additionne deux tableaux élément par élément
    NDarray<T> operator+(const NDarray<T> &other) const;
    // Soustrait deux tableaux élément par élément
    NDarray<T> operator-(const NDarray<T> &other) const;
    // Multiplie deux tableaux élément par élément
    NDarray<T> operator*(const NDarray<T> &other) const;
    // Divise deux tableaux élément par élément
    NDarray<T> operator/(const NDarray<T> &other) const;

    // Opérations avec scalaires
    // Ajoute un scalaire à chaque élément
    NDarray<T> operator+(T scalar) const;
    // Soustrait un scalaire à chaque élément
    NDarray<T> operator-(T scalar) const;
    // Multiplie chaque élément par un scalaire
    NDarray<T> operator*(T scalar) const;
    // Divise chaque élément par un scalaire
    NDarray<T> operator/(T scalar) const;

    // Fonctions statiques pour opérations arithmétiques
    // Additionne deux tableaux (version fonctionnelle)
    static NDarray<T> add(const NDarray<T> &a, const NDarray<T> &b);
    // Soustrait deux tableaux (version fonctionnelle)
    static NDarray<T> subtract(const NDarray<T> &a, const NDarray<T> &b);
    // Multiplie deux tableaux (version fonctionnelle)
    static NDarray<T> multiply(const NDarray<T> &a, const NDarray<T> &b);
    // Divise deux tableaux (version fonctionnelle)
    static NDarray<T> divide(const NDarray<T> &a, const NDarray<T> &b);
    // Effectue un produit matriciel entre deux tableaux 2D
    static NDarray<T> dot(const NDarray<T> &a, const NDarray<T> &b);

    // Accès direct (1D seulement)
    // Accède directement à un élément pour un tableau 1D (version modifiable)
    T &operator[](size_t index) { return data[index]; }
    // Accède directement à un élément pour un tableau 1D (version constante)
    const T &operator[](size_t index) const { return data[index]; }

private:
    // Méthode récursive pour afficher les tableaux multi-dimensionnels
    void print_recursive(size_t dim, size_t start_idx, int indent = 0) const;
    // Calcule l'index plat à partir d'une liste d'indices multi-dimensionnels
    size_t get_flat_index(const std::vector<size_t> &indices) const;
    // Normalise une tranche pour gérer les indices négatifs et vérifier les bornes
    void normalize_slice(Slice &slice, size_t dim_size) const;

    std::vector<T> data;         // Stocke les données du tableau
    std::vector<size_t> shape;   // Stocke les dimensions du tableau
    std::vector<size_t> strides; // Stocke les pas (strides) pour chaque dimension
};

#include "ndarray.tpp"
#endif
