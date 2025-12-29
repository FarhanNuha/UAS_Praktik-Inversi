/****************************************************************************
** Meta object code from reading C++ file 'MethodWidget.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../include/MethodWidget.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MethodWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.4.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
namespace {
struct qt_meta_stringdata_MethodWidget_t {
    uint offsetsAndSizes[22];
    char stringdata0[13];
    char stringdata1[15];
    char stringdata2[1];
    char stringdata3[13];
    char stringdata4[9];
    char stringdata5[18];
    char stringdata6[6];
    char stringdata7[21];
    char stringdata8[22];
    char stringdata9[19];
    char stringdata10[19];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_MethodWidget_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_MethodWidget_t qt_meta_stringdata_MethodWidget = {
    {
        QT_MOC_LITERAL(0, 12),  // "MethodWidget"
        QT_MOC_LITERAL(13, 14),  // "updateGridSize"
        QT_MOC_LITERAL(28, 0),  // ""
        QT_MOC_LITERAL(29, 12),  // "BoundaryData"
        QT_MOC_LITERAL(42, 8),  // "boundary"
        QT_MOC_LITERAL(51, 17),  // "onApproachChanged"
        QT_MOC_LITERAL(69, 5),  // "index"
        QT_MOC_LITERAL(75, 20),  // "onLocalMethodChanged"
        QT_MOC_LITERAL(96, 21),  // "onGlobalMethodChanged"
        QT_MOC_LITERAL(118, 18),  // "onSAVariantChanged"
        QT_MOC_LITERAL(137, 18)   // "onGAVariantChanged"
    },
    "MethodWidget",
    "updateGridSize",
    "",
    "BoundaryData",
    "boundary",
    "onApproachChanged",
    "index",
    "onLocalMethodChanged",
    "onGlobalMethodChanged",
    "onSAVariantChanged",
    "onGAVariantChanged"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_MethodWidget[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   50,    2, 0x0a,    1 /* Public */,
       5,    1,   53,    2, 0x08,    3 /* Private */,
       7,    1,   56,    2, 0x08,    5 /* Private */,
       8,    1,   59,    2, 0x08,    7 /* Private */,
       9,    1,   62,    2, 0x08,    9 /* Private */,
      10,    1,   65,    2, 0x08,   11 /* Private */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    6,

       0        // eod
};

Q_CONSTINIT const QMetaObject MethodWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_MethodWidget.offsetsAndSizes,
    qt_meta_data_MethodWidget,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_MethodWidget_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MethodWidget, std::true_type>,
        // method 'updateGridSize'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const BoundaryData &, std::false_type>,
        // method 'onApproachChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onLocalMethodChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onGlobalMethodChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onSAVariantChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'onGAVariantChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>
    >,
    nullptr
} };

void MethodWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MethodWidget *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->updateGridSize((*reinterpret_cast< std::add_pointer_t<BoundaryData>>(_a[1]))); break;
        case 1: _t->onApproachChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 2: _t->onLocalMethodChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 3: _t->onGlobalMethodChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->onSAVariantChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 5: _t->onGAVariantChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject *MethodWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MethodWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MethodWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int MethodWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 6;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
