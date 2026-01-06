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
    uint offsetsAndSizes[38];
    char stringdata0[13];
    char stringdata1[14];
    char stringdata2[1];
    char stringdata3[9];
    char stringdata4[7];
    char stringdata5[16];
    char stringdata6[14];
    char stringdata7[8];
    char stringdata8[15];
    char stringdata9[13];
    char stringdata10[9];
    char stringdata11[18];
    char stringdata12[6];
    char stringdata13[21];
    char stringdata14[22];
    char stringdata15[19];
    char stringdata16[19];
    char stringdata17[18];
    char stringdata18[16];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_MethodWidget_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_MethodWidget_t qt_meta_stringdata_MethodWidget = {
    {
        QT_MOC_LITERAL(0, 12),  // "MethodWidget"
        QT_MOC_LITERAL(13, 13),  // "methodChanged"
        QT_MOC_LITERAL(27, 0),  // ""
        QT_MOC_LITERAL(28, 8),  // "approach"
        QT_MOC_LITERAL(37, 6),  // "method"
        QT_MOC_LITERAL(44, 15),  // "methodCommitted"
        QT_MOC_LITERAL(60, 13),  // "useMonteCarlo"
        QT_MOC_LITERAL(74, 7),  // "samples"
        QT_MOC_LITERAL(82, 14),  // "updateGridSize"
        QT_MOC_LITERAL(97, 12),  // "BoundaryData"
        QT_MOC_LITERAL(110, 8),  // "boundary"
        QT_MOC_LITERAL(119, 17),  // "onApproachChanged"
        QT_MOC_LITERAL(137, 5),  // "index"
        QT_MOC_LITERAL(143, 20),  // "onLocalMethodChanged"
        QT_MOC_LITERAL(164, 21),  // "onGlobalMethodChanged"
        QT_MOC_LITERAL(186, 18),  // "onSAVariantChanged"
        QT_MOC_LITERAL(205, 18),  // "onGAVariantChanged"
        QT_MOC_LITERAL(224, 17),  // "emitMethodChanged"
        QT_MOC_LITERAL(242, 15)   // "onSelectClicked"
    },
    "MethodWidget",
    "methodChanged",
    "",
    "approach",
    "method",
    "methodCommitted",
    "useMonteCarlo",
    "samples",
    "updateGridSize",
    "BoundaryData",
    "boundary",
    "onApproachChanged",
    "index",
    "onLocalMethodChanged",
    "onGlobalMethodChanged",
    "onSAVariantChanged",
    "onGAVariantChanged",
    "emitMethodChanged",
    "onSelectClicked"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_MethodWidget[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    2,   74,    2, 0x06,    1 /* Public */,
       5,    4,   79,    2, 0x06,    4 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       8,    1,   88,    2, 0x0a,    9 /* Public */,
      11,    1,   91,    2, 0x08,   11 /* Private */,
      13,    1,   94,    2, 0x08,   13 /* Private */,
      14,    1,   97,    2, 0x08,   15 /* Private */,
      15,    1,  100,    2, 0x08,   17 /* Private */,
      16,    1,  103,    2, 0x08,   19 /* Private */,
      17,    0,  106,    2, 0x08,   21 /* Private */,
      18,    0,  107,    2, 0x08,   22 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QString,    3,    4,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::Bool, QMetaType::Int,    3,    4,    6,    7,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 9,   10,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void,
    QMetaType::Void,

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
        // method 'methodChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'methodCommitted'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
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
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'emitMethodChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onSelectClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void MethodWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MethodWidget *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->methodChanged((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 1: _t->methodCommitted((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2])),(*reinterpret_cast< std::add_pointer_t<bool>>(_a[3])),(*reinterpret_cast< std::add_pointer_t<int>>(_a[4]))); break;
        case 2: _t->updateGridSize((*reinterpret_cast< std::add_pointer_t<BoundaryData>>(_a[1]))); break;
        case 3: _t->onApproachChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->onLocalMethodChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 5: _t->onGlobalMethodChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 6: _t->onSAVariantChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 7: _t->onGAVariantChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 8: _t->emitMethodChanged(); break;
        case 9: _t->onSelectClicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MethodWidget::*)(const QString & , const QString & );
            if (_t _q_method = &MethodWidget::methodChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (MethodWidget::*)(const QString & , const QString & , bool , int );
            if (_t _q_method = &MethodWidget::methodCommitted; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
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
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void MethodWidget::methodChanged(const QString & _t1, const QString & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MethodWidget::methodCommitted(const QString & _t1, const QString & _t2, bool _t3, int _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
