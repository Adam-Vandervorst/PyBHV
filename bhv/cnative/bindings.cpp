#include <cstdlib>

#define PY_SSIZE_T_CLEAN size_t

#include <Python.h>
#include "structmember.h"

#include "core.h"

typedef struct {
    PyObject_HEAD
    word_t *data;
} BHV;


static PyObject *BHV_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BHV *v = (BHV *) type->tp_alloc(type, 0);
    v->data = bhv::empty();
    return (PyObject *) v;
}

static void BHV_dealloc(BHV *v) {
    free(v->data);
    Py_TYPE(v)->tp_free((PyObject *) v);
}

static int BHV_init(BHV *v, PyObject *args, PyObject *kwds) {
    return 0;
}

static PyObject *BHV_rand(PyTypeObject *type, PyObject *Py_UNUSED(ignored));

static PyObject *BHV_random(PyTypeObject *type, PyObject *args);

static PyObject *BHV_majority(PyTypeObject *type, PyObject *args);

static PyObject *BHV_threshold(PyTypeObject *type, PyObject *args);

static PyObject *BHV_representative(PyTypeObject *type, PyObject *args);

static PyObject *BHV_window(PyTypeObject *type, PyObject *args);

static PyObject *BHV_select(BHV *cond, PyObject *args);

static PyObject *BHV_ternary(BHV *x, PyObject *args);

static PyObject *BHV_roll_words(BHV *x, PyObject *args);

static PyObject *BHV_roll_word_bits(BHV *x, PyObject *args);

static PyObject *BHV_roll_bits(BHV *x, PyObject *args);

static PyObject *BHV_permute_byte_bits(BHV *x, PyObject *args);

static PyObject *BHV_permute_words(BHV *x, PyObject *args);

static PyObject *BHV_permute(BHV *x, PyObject *args);

static PyObject *BHV_rehash(BHV *x, PyObject *args);

static PyObject *BHV_swap_halves(BHV *x, PyObject *args);

static PyObject *BHV_eq(BHV *v1, PyObject *args);

static PyObject *BHV_richcompare(PyObject *self, PyObject *other, int op) {
    switch (op) {
        case Py_EQ: if (bhv::eq(((BHV*)self)->data, ((BHV*)other)->data)) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        case Py_NE: if (bhv::eq(((BHV*)self)->data, ((BHV*)other)->data)) Py_RETURN_FALSE; else Py_RETURN_TRUE;
        default:
            PyErr_SetString(PyExc_NotImplementedError, "Can only compare BHVs with == and !=");
            return nullptr;
    }
}

static PyObject *BHV_xor(PyObject * v1, PyObject * v2);
static PyObject *BHV_and(PyObject * v1, PyObject * v2);
static PyObject *BHV_or(PyObject * v1, PyObject * v2);
static PyObject *BHV_invert(PyObject * v);

static PyObject *BHV_hamming(BHV * v1, PyObject * args);

static PyObject *BHV_to_bytes(BHV * x, PyObject * Py_UNUSED(ignored));
static PyObject *BHV_from_bytes(PyTypeObject *type, PyObject *args);

static PyObject *__getstate__(BHV *x, PyObject *Py_UNUSED(ignored));
static PyObject *__setstate__(BHV *x, PyObject *args);

static PyObject *BHV_active(BHV * v, PyObject * Py_UNUSED(ignored)) {
    return Py_BuildValue("i", bhv::active(v->data));
}

static PyMemberDef BHV_members[] = {
        {nullptr}
};

static PyMethodDef BHV_methods[] = {
        {"rand",              (PyCFunction) BHV_rand,              METH_CLASS | METH_NOARGS,
                "Bernoulli 1/2 distributed bit vector"},
        {"random",            (PyCFunction) BHV_random,            METH_CLASS | METH_VARARGS,
                "Bernoulli p distributed bit vector"},
        {"majority",          (PyCFunction) BHV_majority,          METH_CLASS | METH_VARARGS,
                "The majority of a list of BHVs"},
        {"threshold",         (PyCFunction) BHV_threshold,         METH_CLASS | METH_VARARGS,
                "Checks if the count is greater than a threshold for a list of BHVs"},
        {"representative",    (PyCFunction) BHV_representative,    METH_CLASS | METH_VARARGS,
                "Random representative of a list of BHVs"},
        {"window",             (PyCFunction) BHV_window,           METH_CLASS | METH_VARARGS,
                "Checks if the count is Between two (inclusive) thresholds of a list of BHVs"},
        {"select",            (PyCFunction) BHV_select,            METH_VARARGS,
                "MUX or IF-THEN-ELSE"},
        {"ternary",           (PyCFunction) BHV_ternary,           METH_VARARGS,
                "Ternary logic operation"},
        {"roll_words",        (PyCFunction) BHV_roll_words,        METH_VARARGS,
                "Word-level rotation"},
        {"roll_word_bits",    (PyCFunction) BHV_roll_word_bits,    METH_VARARGS,
                "Word-level rotation of bits"},
        {"roll_bits",         (PyCFunction) BHV_roll_bits,         METH_VARARGS,
                "A proper roll of bits"},
        {"permute_byte_bits", (PyCFunction) BHV_permute_byte_bits, METH_VARARGS,
                "Permutes the bits of every byte"},
        {"permute_words",     (PyCFunction) BHV_permute_words,     METH_VARARGS,
                "Word-level permutation"},
        {"permute",           (PyCFunction) BHV_permute,           METH_VARARGS,
                "Default permutation"},
        {"rehash",            (PyCFunction) BHV_rehash,            METH_NOARGS,
                "Hash the vector into another vector"},
        {"swap_halves",       (PyCFunction) BHV_swap_halves,       METH_NOARGS,
                "Swap the halves of the vector"},
        {"eq",                (PyCFunction) BHV_eq,                METH_VARARGS,
                "Check equality"},
        {"hamming",           (PyCFunction) BHV_hamming,           METH_VARARGS,
                "Hamming distance between two BHVs"},
        {"active",            (PyCFunction) BHV_active,            METH_NOARGS,
                "Count the number of active bits"},
        {"to_bytes",          (PyCFunction) BHV_to_bytes,          METH_NOARGS,
                "Bytes normalized form"},
        {"from_bytes",        (PyCFunction) BHV_from_bytes,        METH_CLASS | METH_VARARGS,
                "Construct from bytes normalized form"},
        {"__getstate__",      (PyCFunction) __getstate__,          METH_NOARGS,
                "Pickle the vector"},
        {"__setstate__",      (PyCFunction) __setstate__,          METH_VARARGS,
                "Un-pickle the vector"},
        {nullptr}
};

static PyNumberMethods BHV_nb_methods = {
        .nb_invert = (unaryfunc) BHV_invert,
        .nb_and = (binaryfunc) BHV_and,
        .nb_xor = (binaryfunc) BHV_xor,
        .nb_or = (binaryfunc) BHV_or,
};

static PyTypeObject BHVType = {
        .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
        .tp_name = "bhv.cnative.CNativePackedBHV",
        .tp_basicsize = sizeof(BHV),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) BHV_dealloc,
        .tp_as_number = &BHV_nb_methods,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Packed, native implementation of Boolean Hypervectors",
        .tp_richcompare = BHV_richcompare,
        .tp_methods = BHV_methods,
        .tp_members = BHV_members,
        .tp_dict = PyDict_New(),
        .tp_init = (initproc) BHV_init,
        .tp_new = BHV_new,
};


static PyObject *BHV_rand(PyTypeObject *type, PyObject *Py_UNUSED(ignored)) {
    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::rand_into(((BHV *) v)->data);
    return v;
}

static PyObject *BHV_random(PyTypeObject *type, PyObject *args) {
    float p;
    if (!PyArg_ParseTuple(args, "f", &p))
        return nullptr;

    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::random_into(((BHV *) v)->data, p);
    return v;
}

static PyObject *BHV_hamming(BHV * v1, PyObject * args) {
    BHV * v2;
    if (!PyArg_ParseTuple(args, "O!", &BHVType, &v2))
        return nullptr;

    return Py_BuildValue("i", bhv::hamming(v1->data, v2->data));
}

static PyObject *BHV_majority(PyTypeObject *type, PyObject *args) {
    PyObject * vector_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &vector_list))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);
    size_t even = 1 - n_vectors % 2;

    word_t **vs = (word_t **) malloc((n_vectors + even) * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    if (even) vs[n_vectors] = bhv::rand();

    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::true_majority_into(vs, n_vectors + even, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_threshold(PyTypeObject *type, PyObject *args) {
    PyObject * vector_list;
    int32_t t;

    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &vector_list, &t))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);

    word_t **vs = (word_t **) malloc((n_vectors) * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::threshold_into(vs, n_vectors, t, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_representative(PyTypeObject *type, PyObject *args) {
    PyObject * vector_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &vector_list))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);

    word_t **vs = (word_t **) malloc(n_vectors * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    PyObject * ret = BHV_new(type, nullptr, nullptr);
    bhv::representative_into(vs, n_vectors, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_window(PyTypeObject *type, PyObject *args) {
    PyObject * vector_list;
    int32_t b;
    int32_t t;

    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &vector_list, &b, &t))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);

    word_t **vs = (word_t **) malloc((n_vectors) * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::window_into(vs, n_vectors, b, t, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_xor(PyObject * v1, PyObject * v2) {
    if (not PyObject_IsInstance(v1, (PyObject *) &BHVType) or
        not PyObject_IsInstance(v2, (PyObject *) &BHVType)) {
        PyErr_SetString(PyExc_TypeError, "Only BHV argument(s) supported");
        return nullptr;
    }

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::xor_into(((BHV *) v1)->data, ((BHV *) v2)->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_and(PyObject * v1, PyObject * v2) {
    if (not PyObject_IsInstance(v1, (PyObject *) &BHVType) or
        not PyObject_IsInstance(v2, (PyObject *) &BHVType)) {
        PyErr_SetString(PyExc_TypeError, "Only BHV argument(s) supported");
        return nullptr;
    }

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::and_into(((BHV *) v1)->data, ((BHV *) v2)->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_or(PyObject * v1, PyObject * v2) {
    if (not PyObject_IsInstance(v1, (PyObject *) &BHVType) or
        not PyObject_IsInstance(v2, (PyObject *) &BHVType)) {
        PyErr_SetString(PyExc_TypeError, "Only BHV argument(s) supported");
        return nullptr;
    }

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::or_into(((BHV *) v1)->data, ((BHV *) v2)->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_invert(PyObject * v) {
    if (not PyObject_IsInstance(v, (PyObject *) &BHVType)) {
        PyErr_SetString(PyExc_TypeError, "Only BHV argument(s) supported");
        return nullptr;
    }

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::invert_into(((BHV *) v)->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_eq(BHV * v1, PyObject * args) {
    BHV * v2;

    if (!PyArg_ParseTuple(args, "O!", &BHVType, &v2))
        return nullptr;

    if (bhv::eq(v1->data, v2->data)) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}

static PyObject *BHV_select(BHV * cond, PyObject * args) {
    BHV * when1;
    BHV * when0;

    if (!PyArg_ParseTuple(args, "O!O!", &BHVType, &when1, &BHVType, &when0))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::select_into(cond->data, when1->data, when0->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_ternary(BHV * x, PyObject * args) {
    BHV * y;
    BHV * z;
    uint8_t op;

    if (!PyArg_ParseTuple(args, "O!O!b", &BHVType, &y, &BHVType, &z, &op))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::dynamic_ternary_into(x->data, y->data, z->data, ((BHV *) ret)->data, op);
    return ret;
}

static PyObject *BHV_roll_words(BHV * x, PyObject * args) {
    int32_t p;
    if (!PyArg_ParseTuple(args, "i", &p))
        return nullptr;

    PyObject * v = BHV_new(&BHVType, nullptr, nullptr);
    bhv::roll_words_into(x->data, p, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_roll_word_bits(BHV * x, PyObject * args) {
    int32_t p;
    if (!PyArg_ParseTuple(args, "i", &p))
        return nullptr;

    PyObject * v = BHV_new(&BHVType, nullptr, nullptr);
    bhv::roll_word_bits_into(x->data, p, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_roll_bits(BHV * x, PyObject * args) {
    int32_t p;
    if (!PyArg_ParseTuple(args, "i", &p))
        return nullptr;

    PyObject * v = BHV_new(&BHVType, nullptr, nullptr);
    bhv::roll_bits_into(x->data, p, ((BHV *) v)->data);
    return v;
}

static PyObject *BHV_permute_words(BHV * x, PyObject * args) {
    int32_t perm;

    if (!PyArg_ParseTuple(args, "i", &perm))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::permute_words_into(x->data, perm, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_permute_byte_bits(BHV * x, PyObject * args) {
    int32_t perm;

    if (!PyArg_ParseTuple(args, "i", &perm))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::permute_byte_bits_into(x->data, perm, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_permute(BHV * x, PyObject * args) {
    int32_t perm;

    if (!PyArg_ParseTuple(args, "i", &perm))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::permute_into(x->data, perm, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_rehash(BHV * x, PyObject * args) {
    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    ((BHV *) ret)->data = bhv::zero();
    bhv::rehash_into(x->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_swap_halves(BHV * x, PyObject * args) {
    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::swap_halves_into(x->data, ((BHV *) ret)->data);
    return ret;
}

static PyObject *BHV_to_bytes(BHV * x, PyObject * Py_UNUSED(ignored)) {
    return PyBytes_FromStringAndSize((char*)x->data, BYTES);
}

static PyObject *BHV_from_bytes(PyTypeObject *type, PyObject *args) {
    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    char *buf;
    size_t size;

    if (!PyArg_ParseTuple(args, "s#", &buf, &size))
        return nullptr;

    if (size != BYTES) {
        PyErr_SetString(PyExc_TypeError, "Bytes object didn't have the right size");
        return nullptr;
    }

    memcpy(((BHV *) ret)->data, buf, BYTES);
    return ret;
}

static PyObject *__getstate__(BHV *x, PyObject *Py_UNUSED(ignored)) {
    return PyBytes_FromStringAndSize((char*)x->data, BYTES);
}

static PyObject *__setstate__(BHV *x, PyObject *args) {
    char *buf;
    size_t size;

    if (!PyArg_ParseTuple(args, "s#", &buf, &size))
        return nullptr;

    if (size != BYTES) {
        PyErr_SetString(PyExc_TypeError, "Bytes object didn't have the right size");
        return nullptr;
    }

    memcpy(x->data, buf, BYTES);
    Py_RETURN_NONE;
}

static PyObject *dimension(PyObject * self, PyObject * args, PyObject * kwds) {
    return PyLong_FromLong(BITS);
}


static PyMethodDef module_methods[] = {
        {"_DIMENSION", (PyCFunction) dimension, METH_NOARGS,
                "return vector with the library was compiled for"},
        {nullptr}
};

#ifndef PyMODINIT_FUNC    /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif


static struct PyModuleDef cModPyDem = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "cnative",
        .m_doc = "",
        .m_size = -1,
        .m_methods = module_methods
};


PyMODINIT_FUNC PyInit_cnative(void) {
    PyObject * m;
    if (PyType_Ready(&BHVType) < 0)
        return nullptr;

    m = PyModule_Create(&cModPyDem);
    if (m == nullptr)
        return nullptr;

    BHV * z = (BHV *) BHVType.tp_alloc(&BHVType, 0);
    z->data = bhv::zero();
    PyDict_SetItemString(BHVType.tp_dict, "ZERO", (PyObject *) z);
    BHV * o = (BHV *) BHVType.tp_alloc(&BHVType, 0);
    o->data = bhv::one();
    PyDict_SetItemString(BHVType.tp_dict, "ONE", (PyObject *) o);
    BHV * h = (BHV *) BHVType.tp_alloc(&BHVType, 0);
    h->data = bhv::half();
    PyDict_SetItemString(BHVType.tp_dict, "HALF", (PyObject *) h);

    Py_INCREF(&BHVType);
    PyModule_AddObject(m, "CNativePackedBHV", (PyObject *) &BHVType);

    return m;
};


int main() {
    return 0;
}
