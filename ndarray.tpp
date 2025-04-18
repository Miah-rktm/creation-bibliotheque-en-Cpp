#include "ndarray.h"
#include <functional>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <iostream>

template <typename T>
NDarray<T>::NDarray(std::initializer_list<size_t> dims, T value)
{
    shape = dims;                 // Définit la forme du tableau
    size_t tsize = 1;             // Calcule la taille totale
    strides.resize(shape.size()); // Redimensionne le vecteur des pas
    for (size_t i = shape.size(); i > 0; --i)
    {
        strides[i - 1] = tsize; // Calcule les pas pour chaque dimension
        tsize *= shape[i - 1];  // Met à jour la taille totale
    }
    data.resize(tsize, value); // Initialise les données avec la valeur donnée
}

// Constructeur avec vecteur pour les dimensions
template <typename T>
NDarray<T>::NDarray(std::vector<size_t> dims, T value) : shape(dims)
{
    size_t tsize = 1;             // Calcule la taille totale
    strides.resize(shape.size()); // Redimensionne le vecteur des pas
    for (size_t i = shape.size(); i > 0; --i)
    {
        strides[i - 1] = tsize; // Calcule les pas pour chaque dimension
        tsize *= shape[i - 1];  // Met à jour la taille totale
    }
    data.resize(tsize, value); // Initialise les données avec la valeur donnée
}

// Retourne la taille totale du tableau
template <typename T>
size_t NDarray<T>::getSize() const
{
    return data.size();
}

// Calcule l'index plat à partir d'une liste d'indices multi-dimensionnels
template <typename T>
size_t NDarray<T>::get_flat_index(const std::vector<size_t> &indices) const
{
    if (indices.size() != shape.size())
    {
        throw std::out_of_range("Le nombre d'indices ne correspond pas aux dimensions du tableau");
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] >= shape[i])
        {
            throw std::out_of_range("Index hors des limites");
        }
        index += indices[i] * strides[i]; // Calcule l'index plat en utilisant les pas
    }
    return index;
}

// Normalise une tranche pour gérer les indices négatifs et vérifier les bornes
template <typename T>
void NDarray<T>::normalize_slice(Slice &slice, size_t dim_size) const
{
    // Convertir les indices négatifs en indices positifs
    if (slice.start < 0)
        slice.start += dim_size;
    if (slice.stop < 0)
        slice.stop += dim_size;

    // Vérifier les bornes pour l'index de départ
    if (slice.start < 0 || slice.start >= static_cast<long long>(dim_size))
    {
        throw std::out_of_range("Index de départ de la tranche hors des limites");
    }
    // Permettre stop = -1 ou stop = dim_size pour pas négatif (exclusif)
    if (slice.stop < -1 || slice.stop > static_cast<long long>(dim_size))
    {
        throw std::out_of_range("Index d'arrêt de la tranche hors des limites");
    }

    if (slice.step == 0)
    {
        throw std::invalid_argument("Le pas de la tranche ne peut pas être zéro");
    }

    // Pour les pas négatifs, pas d'échange automatique de start et stop
    // La logique est gérée dans operator[]
}

// Affiche le tableau dans un format lisible
template <typename T>
void NDarray<T>::print() const
{
    std::cout << "array(";
    if (shape.empty())
    {
        std::cout << data[0]; // Cas d'un tableau scalaire
    }
    else
    {
        print_recursive(0, 0); // Appelle la méthode récursive pour l'affichage
    }
    std::cout << ")" << std::endl;
}

// Affiche récursivement les tableaux multi-dimensionnels
template <typename T>
void NDarray<T>::print_recursive(size_t dim, size_t start_idx, int indent) const
{
    if (dim == shape.size() - 1)
    {
        std::cout << "["; // Début d'une ligne
        for (size_t i = 0; i < shape[dim]; ++i)
        {
            std::cout << data[start_idx + i * strides[dim]]; // Affiche chaque élément
            if (i < shape[dim] - 1)
                std::cout << ", "; // Ajoute une virgule entre les éléments
        }
        std::cout << "]"; // Fin de la ligne
    }
    else
    {
        std::cout << "[\n"; // Début d'un tableau imbriqué
        for (size_t i = 0; i < shape[dim]; ++i)
        {
            print_recursive(dim + 1, start_idx + i * strides[dim], indent + 2); // Appel récursif
            if (i < shape[dim] - 1)
                std::cout << "\n"; // Saut de ligne entre sous-tableaux
        }
        std::cout << "\n]"; // Fin du tableau imbriqué
    }
}

// Accède à un élément spécifique (version modifiable)
template <typename T>
T &NDarray<T>::at(std::initializer_list<size_t> indices)
{
    std::vector<size_t> idx_vec(indices.begin(), indices.end());
    return data[get_flat_index(idx_vec)]; // Retourne une référence à l'élément
}

// Accède à un élément spécifique (version constante)
template <typename T>
const T &NDarray<T>::at(std::initializer_list<size_t> indices) const
{
    std::vector<size_t> idx_vec(indices.begin(), indices.end());
    return data[get_flat_index(idx_vec)]; // Retourne une référence constante à l'élément
}

// Extrait une sous-partie du tableau via des tranches
template <typename T>
NDarray<T> NDarray<T>::operator[](const std::vector<Slice> &slices) const
{
    if (slices.size() > shape.size())
        throw std::out_of_range("Trop de tranches pour les dimensions du tableau");

    std::vector<Slice> norm_slices = slices; // Copie des tranches
    std::vector<size_t> new_shape;           // Nouvelle forme du tableau résultant
    for (size_t i = 0; i < slices.size(); ++i)
    {
        norm_slices[i] = slices[i];
        normalize_slice(norm_slices[i], shape[i]); // Normalise chaque tranche

        long long slice_size;
        if (norm_slices[i].step > 0)
        {
            // Pour pas positif, stop est exclusif
            slice_size = (norm_slices[i].stop > norm_slices[i].start)
                             ? (norm_slices[i].stop - norm_slices[i].start + norm_slices[i].step - 1) / norm_slices[i].step
                             : 0;
        }
        else
        {
            // Pour pas négatif, stop est exclusif
            long long effective_stop = norm_slices[i].stop >= 0 ? norm_slices[i].stop : -1;
            slice_size = (norm_slices[i].start > effective_stop)
                             ? (norm_slices[i].start - effective_stop - norm_slices[i].step - 1) / (-norm_slices[i].step)
                             : 0;
        }
        if (slice_size < 0)
            throw std::invalid_argument("Plage de tranche invalide (taille négative)");
        new_shape.push_back(static_cast<size_t>(slice_size)); // Ajoute la taille de la tranche
    }
    // Compléter avec des tranches complètes pour les dimensions restantes
    for (size_t i = slices.size(); i < shape.size(); ++i)
    {
        norm_slices.emplace_back(0, shape[i], 1);
        new_shape.push_back(shape[i]);
    }

    NDarray<T> result(new_shape);                      // Crée le tableau résultant
    std::vector<size_t> counters(new_shape.size(), 0); // Compteurs pour l'itération
    bool done = false;
    while (!done)
    {
        std::vector<size_t> src_indices;
        for (size_t i = 0; i < counters.size(); ++i)
        {
            long long val = norm_slices[i].start + counters[i] * norm_slices[i].step; // Calcule l'index source
            if (val < 0 || val >= static_cast<long long>(shape[i]))
                throw std::out_of_range("Index calculé hors des limites pendant le slicing : val=" + std::to_string(val));
            src_indices.push_back(static_cast<size_t>(val));
        }

        result.data[result.get_flat_index(counters)] = data[get_flat_index(src_indices)]; // Copie la valeur

        // Incrémentation multi-dimensionnelle
        for (int i = counters.size() - 1; i >= 0; --i)
        {
            counters[i]++;
            if (counters[i] < new_shape[i])
                break;
            counters[i] = 0;
            if (i == 0)
                done = true;
        }
    }

    return result;
}
template <typename T>
NDarray<T> NDarray<T>::zeros(std::initializer_list<size_t> dims)
{
    return NDarray<T>(dims, 0);
}
template <typename T>
NDarray<T> NDarray<T>::ones(std::initializer_list<size_t> dims)
{
    return NDarray<T>(dims, 1);
}
template <typename T>
NDarray<T> NDarray<T>::full(std::initializer_list<size_t> dims, T value)
{
    return NDarray<T>(dims, value);
}

template <typename T>
NDarray<T> NDarray<T>::eye(size_t n)
{
    NDarray<T> result({n, n}, 0);
    for (size_t i = 0; i < n; ++i)
    {
        result.at({i, i}) = 1; // Place des 1 sur la diagonale
    }
    return result;
}

// Crée un tableau 1D avec une séquence linéaire
template <typename T>
NDarray<T> NDarray<T>::arange(T start, T stop, T step)
{
    if (step == 0)
        throw std::invalid_argument("Le pas ne peut pas être zéro");
    size_t size = static_cast<size_t>((stop - start) / step);
    if ((stop - start) * step < 0)
        size = 0; // Ajuste la taille si la plage est vide
    NDarray<T> arr({size});
    for (size_t i = 0; i < size; ++i)
    {
        arr.data[i] = start + i * step; // Remplit avec la séquence
    }
    return arr;
}

// Crée un tableau 1D avec des valeurs équidistantes
template <typename T>
NDarray<T> NDarray<T>::linspace(T start, T stop, size_t num)
{
    if (num == 0)
        throw std::invalid_argument("Le nombre d'échantillons doit être positif");
    if (num == 1)
        return NDarray<T>({1}, start);
    NDarray<T> arr({num});
    T step = (stop - start) / (num - 1); // Calcule le pas
    for (size_t i = 0; i < num; ++i)
    {
        arr.data[i] = start + i * step; // Remplit avec les valeurs
    }
    return arr;
}

// Crée un tableau avec des valeurs aléatoires
template <typename T>
NDarray<T> NDarray<T>::rand(std::initializer_list<size_t> dims, T min_val, T max_val)
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "T doit être un type entier ou à virgule flottante");
    NDarray<T> result(dims);
    static std::random_device rd;
    static std::mt19937 gen(rd()); // Générateur aléatoire
    if constexpr (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < result.getSize(); ++i)
        {
            result.data[i] = dist(gen); // Remplit avec des entiers aléatoires
        }
    }
    else if constexpr (std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < result.getSize(); ++i)
        {
            result.data[i] = dist(gen); // Remplit avec des flottants aléatoires
        }
    }
    return result;
}

// Crée un tableau d'entiers aléatoires dans l'intervalle [low, high)
template <typename T>
NDarray<T> NDarray<T>::randint(T low, T high, std::initializer_list<size_t> dims)
{
    static_assert(std::is_integral<T>::value, "randint ne supporte que les types entiers");
    if (low >= high)
        throw std::invalid_argument("low doit être inférieur à high");
    NDarray<T> result(dims);
    static std::random_device rd;
    static std::mt19937 gen(rd());                        // Générateur aléatoire
    std::uniform_int_distribution<T> dist(low, high - 1); // high exclusif
    for (size_t i = 0; i < result.getSize(); ++i)
    {
        result.data[i] = dist(gen); // Remplit avec des entiers aléatoires
    }
    return result;
}

// Reformate le tableau à une nouvelle forme
template <typename T>
void NDarray<T>::reshape(std::initializer_list<size_t> new_shape)
{
    size_t new_size = 1;
    for (size_t d : new_shape)
    {
        new_size *= d; // Calcule la taille totale de la nouvelle forme
    }
    if (new_size != getSize())
    {
        throw std::runtime_error("La nouvelle forme est incompatible avec la taille actuelle");
    }
    shape = new_shape;
    strides.resize(shape.size());
    size_t tsize = 1;
    for (size_t i = shape.size(); i > 0; --i)
    {
        strides[i - 1] = tsize; // Recalcule les pas
        tsize *= shape[i - 1];
    }
}

// Aplatit le tableau en un tableau 1D
template <typename T>
NDarray<T> NDarray<T>::flatten() const
{
    NDarray<T> flat({getSize()});
    flat.data = data; // Copie les données
    return flat;
}

// Concatène deux tableaux le long d'un axe
template <typename T>
NDarray<T> NDarray<T>::concatenate(const NDarray<T> &other, size_t axis)
{
    if (shape.size() != other.shape.size())
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de dimensions");
    }
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i != axis && shape[i] != other.shape[i])
        {
            throw std::runtime_error("Formes incompatibles le long de l'axe non concaténé");
        }
    }
    std::vector<size_t> new_shape = shape;
    new_shape[axis] += other.shape[axis]; // Ajuste la dimension de l'axe
    NDarray<T> result(new_shape);
    size_t offset = 0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[offset++] = data[i]; // Copie les données du premier tableau
    }
    for (size_t i = 0; i < other.data.size(); ++i)
    {
        result.data[offset++] = other.data[i]; // Copie les données du second tableau
    }
    return result;
}

// Concatène deux tableaux horizontalement
template <typename T>
NDarray<T> NDarray<T>::hstack(const NDarray<T> &other)
{
    std::vector<size_t> shape1 = shape;
    std::vector<size_t> shape2 = other.shape;
    if (shape1.size() == 1 && shape2.size() == 1)
    {
        return concatenate(other, 0); // Cas 1D
    }
    if (shape1[0] != shape2[0])
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de lignes pour hstack");
    }
    return concatenate(other, 1); // Concaténation le long de l'axe 1
}

// Concatène deux tableaux verticalement
template <typename T>
NDarray<T> NDarray<T>::vstack(const NDarray<T> &other)
{
    NDarray<T> arr1_2d = *this;
    NDarray<T> arr2_2d = other;
    if (shape.size() == 1)
    {
        arr1_2d.reshape({1, shape[0]}); // Convertit 1D en 2D
    }
    if (other.shape.size() == 1)
    {
        arr2_2d.reshape({1, other.shape[0]}); // Convertit 1D en 2D
    }
    if (arr1_2d.shape[1] != arr2_2d.shape[1])
    {
        throw std::runtime_error("Les tableaux doivent avoir le même nombre de colonnes pour vstack");
    }
    return arr1_2d.concatenate(arr2_2d, 0); // Concaténation le long de l'axe 0
}

// Additionne deux tableaux élément par élément
template <typename T>
NDarray<T> NDarray<T>::operator+(const NDarray<T> &other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Les formes doivent correspondre pour l'addition");
    }
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// Soustrait deux tableaux élément par élément
template <typename T>
NDarray<T> NDarray<T>::operator-(const NDarray<T> &other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Les formes doivent correspondre pour la soustraction");
    }
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Multiplie deux tableaux élément par élément
template <typename T>
NDarray<T> NDarray<T>::operator*(const NDarray<T> &other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Les formes doivent correspondre pour la multiplication");
    }
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

// Divise deux tableaux élément par élément
template <typename T>
NDarray<T> NDarray<T>::operator/(const NDarray<T> &other) const
{
    if (shape != other.shape)
    {
        throw std::invalid_argument("Les formes doivent correspondre pour la division");
    }
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (other.data[i] == 0)
            throw std::runtime_error("Division par zéro détectée");
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

// Ajoute un scalaire à chaque élément
template <typename T>
NDarray<T> NDarray<T>::operator+(T scalar) const
{
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

// Soustrait un scalaire à chaque élément
template <typename T>
NDarray<T> NDarray<T>::operator-(T scalar) const
{
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

// Multiplie chaque élément par un scalaire
template <typename T>
NDarray<T> NDarray<T>::operator*(T scalar) const
{
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Divise chaque élément par un scalaire
template <typename T>
NDarray<T> NDarray<T>::operator/(T scalar) const
{
    if (scalar == 0)
        throw std::runtime_error("Division par zéro détectée");
    NDarray<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

// Additionne deux tableaux (version fonctionnelle)
template <typename T>
NDarray<T> NDarray<T>::add(const NDarray<T> &a, const NDarray<T> &b)
{
    return a + b;
}

// Soustrait deux tableaux (version fonctionnelle)
template <typename T>
NDarray<T> NDarray<T>::subtract(const NDarray<T> &a, const NDarray<T> &b)
{
    return a - b;
}

// Multiplie deux tableaux (version fonctionnelle)
template <typename T>
NDarray<T> NDarray<T>::multiply(const NDarray<T> &a, const NDarray<T> &b)
{
    return a * b;
}

// Divise deux tableaux (version fonctionnelle)
template <typename T>
NDarray<T> NDarray<T>::divide(const NDarray<T> &a, const NDarray<T> &b)
{
    return a / b;
}

// Effectue un produit matriciel entre deux tableaux 2D
template <typename T>
NDarray<T> NDarray<T>::dot(const NDarray<T> &a, const NDarray<T> &b)
{
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw std::invalid_argument("Le produit matriciel n'est supporté que pour les tableaux 2D");
    if (a.shape[1] != b.shape[0])
        throw std::invalid_argument("Les dimensions internes doivent correspondre pour le produit matriciel");

    NDarray<T> result({a.shape[0], b.shape[1]}, 0);
    for (size_t i = 0; i < a.shape[0]; ++i)
    {
        for (size_t j = 0; j < b.shape[1]; ++j)
        {
            for (size_t k = 0; k < a.shape[1]; ++k)
            {
                result.data[i * result.strides[0] + j] +=
                    a.data[i * a.strides[0] + k] * b.data[k * b.strides[0] + j]; // Calcule le produit
            }
        }
    }
    return result;
}
