#include <cstdlib>

#include <Python.h>
#include "structmember.h"

#include "packed.h"

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

static PyObject *BHV_true_majority(PyTypeObject *type, PyObject *args);

static PyObject *BHV_representative(PyTypeObject *type, PyObject *args);

static PyObject *BHV_eq(BHV *v1, PyObject *args);

static PyObject *BHV_xor(BHV *v1, PyObject *args);

static PyObject *BHV_hamming(BHV *v1, PyObject *args);

static PyObject *BHV_active(BHV *v, PyObject *Py_UNUSED(ignored)) {
    return Py_BuildValue("i", bhv::active(v->data));
}

static PyMemberDef BHV_members[] = {
        {nullptr}
};

static PyMethodDef BHV_methods[] = {
        {"rand",           (PyCFunction) BHV_rand,       METH_CLASS | METH_NOARGS,
                "Set vector elements to random values"},
        {"_true_majority", (PyCFunction) BHV_true_majority,  METH_CLASS | METH_VARARGS,
                "The majority of a list of BHVs"},
        {"representative", (PyCFunction) BHV_representative, METH_CLASS | METH_VARARGS,
                "Random representative of a list of BHVs"},
        {"__eq__",         (PyCFunction) BHV_eq,         METH_VARARGS,
                "Check equality"},
        {"hamming",         (PyCFunction) BHV_hamming,         METH_VARARGS,
                "Hamming distance between two BHVs"},
        {"active",         (PyCFunction) BHV_active,     METH_NOARGS,
                "Count the number of active bits"},
        {"__xor__",        (PyCFunction) BHV_xor,        METH_VARARGS,
                "XOR of two BHVs"},
        {nullptr}
};


static PyTypeObject BHVType = {
        PyVarObject_HEAD_INIT(nullptr, 0)
        "bhv.NativePackedBHV",
        sizeof(BHV),
        0,
        (destructor) BHV_dealloc,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
//        (reprfunc)BHV_repr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,
        "Packed, native implementation of Boolean Hypervectors",
        nullptr,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
        BHV_methods,
        BHV_members,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        (initproc) BHV_init,
        nullptr,
        BHV_new,
};


static PyObject *BHV_rand(PyTypeObject *type, PyObject *Py_UNUSED(ignored)) {
    PyObject * v = BHV_new(type, nullptr, nullptr);
    bhv::rand_into(((BHV *) v)->data);
    return v;
}

static PyObject *BHV_hamming(BHV *v1, PyObject *args) {
    BHV *v2;
    if (!PyArg_ParseTuple(args, "O!", &BHVType, &v2))
        return nullptr;

    return Py_BuildValue("i", bhv::hamming(v1->data, v2->data));
}

static PyObject *BHV_true_majority(PyTypeObject *type, PyObject *args) {
    PyObject * vector_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &vector_list))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);

    word_t **vs = (word_t **) malloc(n_vectors * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    PyObject * ret = type->tp_alloc(type, 0);
    ((BHV *) ret)->data = bhv::true_majority(vs, n_vectors);
    return ret;
}

static PyObject *BHV_representative(PyTypeObject *type, PyObject *args) {
    // TODO
    PyObject * vector_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &vector_list))
        return nullptr;

    size_t n_vectors = PyList_GET_SIZE(vector_list);

    word_t **vs = (word_t **) malloc(n_vectors * sizeof(word_t *));

    for (size_t i = 0; i < n_vectors; ++i) {
        PyObject * v_i_py = PyList_GetItem(vector_list, i);
        vs[i] = ((BHV *) v_i_py)->data;
    }

    PyObject * ret = type->tp_alloc(type, 0);
    ((BHV *) ret)->data = bhv::representative(vs, n_vectors);
    return ret;
}

static PyObject *BHV_xor(BHV *v1, PyObject *args) {
    BHV *v2;
    if (!PyArg_ParseTuple(args, "O!", &BHVType, &v2))
        return nullptr;

    PyObject * ret = BHV_new(&BHVType, nullptr, nullptr);
    bhv::xor_into(v1->data, v2->data, ((BHV *) ret)->data);
    return ret;
}


static PyObject *BHV_eq(BHV *v1, PyObject *args) {
    BHV *v2;

    if (!PyArg_ParseTuple(args, "O!", &BHVType, &v2))
        return nullptr;

    if (bhv::eq(v1->data, v2->data)) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
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
        PyModuleDef_HEAD_INIT,
        "native",
        "",
        -1,
        module_methods
};


PyMODINIT_FUNC PyInit_native(void) {
    PyObject * m;
    if (PyType_Ready(&BHVType) < 0)
        return nullptr;

    m = PyModule_Create(&cModPyDem);
    if (m == nullptr)
        return nullptr;

    Py_INCREF(&BHVType);
    PyModule_AddObject(m, "NativePackedBHV", (PyObject *) &BHVType);

    return m;
};


int main() {
    return 0;
}
