/****************************************************************************
** Meta object code from reading C++ file 'MapViewer3D.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../include/MapViewer3D.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MapViewer3D.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_MapViewer3D_t {
    uint offsetsAndSizes[16];
    char stringdata0[12];
    char stringdata1[15];
    char stringdata2[1];
    char stringdata3[13];
    char stringdata4[9];
    char stringdata5[15];
    char stringdata6[19];
    char stringdata7[9];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_MapViewer3D_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_MapViewer3D_t qt_meta_stringdata_MapViewer3D = {
    {
        QT_MOC_LITERAL(0, 11),  // "MapViewer3D"
        QT_MOC_LITERAL(12, 14),  // "updateBoundary"
        QT_MOC_LITERAL(27, 0),  // ""
        QT_MOC_LITERAL(28, 12),  // "BoundaryData"
        QT_MOC_LITERAL(41, 8),  // "boundary"
        QT_MOC_LITERAL(50, 14),  // "updateStations"
        QT_MOC_LITERAL(65, 18),  // "QList<StationData>"
        QT_MOC_LITERAL(84, 8)   // "stations"
    },
    "MapViewer3D",
    "updateBoundary",
    "",
    "BoundaryData",
    "boundary",
    "updateStations",
    "QList<StationData>",
    "stations"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_MapViewer3D[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   26,    2, 0x0a,    1 /* Public */,
       5,    1,   29,    2, 0x0a,    3 /* Public */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 6,    7,

       0        // eod
};

Q_CONSTINIT const QMetaObject MapViewer3D::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_MapViewer3D.offsetsAndSizes,
    qt_meta_data_MapViewer3D,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_MapViewer3D_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MapViewer3D, std::true_type>,
        // method 'updateBoundary'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const BoundaryData &, std::false_type>,
        // method 'updateStations'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVector<StationData> &, std::false_type>
    >,
    nullptr
} };

void MapViewer3D::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MapViewer3D *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->updateBoundary((*reinterpret_cast< std::add_pointer_t<BoundaryData>>(_a[1]))); break;
        case 1: _t->updateStations((*reinterpret_cast< std::add_pointer_t<QList<StationData>>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject *MapViewer3D::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MapViewer3D::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MapViewer3D.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int MapViewer3D::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 2;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
