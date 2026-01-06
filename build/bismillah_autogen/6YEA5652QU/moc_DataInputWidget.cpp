/****************************************************************************
** Meta object code from reading C++ file 'DataInputWidget.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../include/DataInputWidget.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'DataInputWidget.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_DataInputWidget_t {
    uint offsetsAndSizes[42];
    char stringdata0[16];
    char stringdata1[15];
    char stringdata2[1];
    char stringdata3[19];
    char stringdata4[9];
    char stringdata5[21];
    char stringdata6[20];
    char stringdata7[11];
    char stringdata8[17];
    char stringdata9[7];
    char stringdata10[15];
    char stringdata11[9];
    char stringdata12[12];
    char stringdata13[17];
    char stringdata14[19];
    char stringdata15[22];
    char stringdata16[8];
    char stringdata17[8];
    char stringdata18[22];
    char stringdata19[19];
    char stringdata20[6];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_DataInputWidget_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_DataInputWidget_t qt_meta_stringdata_DataInputWidget = {
    {
        QT_MOC_LITERAL(0, 15),  // "DataInputWidget"
        QT_MOC_LITERAL(16, 14),  // "stationsLoaded"
        QT_MOC_LITERAL(31, 0),  // ""
        QT_MOC_LITERAL(32, 18),  // "QList<StationData>"
        QT_MOC_LITERAL(51, 8),  // "stations"
        QT_MOC_LITERAL(60, 20),  // "computationRequested"
        QT_MOC_LITERAL(81, 19),  // "computationComplete"
        QT_MOC_LITERAL(101, 10),  // "resultText"
        QT_MOC_LITERAL(112, 16),  // "HypocenterResult"
        QT_MOC_LITERAL(129, 6),  // "result"
        QT_MOC_LITERAL(136, 14),  // "onLoadDataFile"
        QT_MOC_LITERAL(151, 8),  // "onAddRow"
        QT_MOC_LITERAL(160, 11),  // "onDeleteRow"
        QT_MOC_LITERAL(172, 16),  // "onComputeClicked"
        QT_MOC_LITERAL(189, 18),  // "onTableDataChanged"
        QT_MOC_LITERAL(208, 21),  // "onComputationProgress"
        QT_MOC_LITERAL(230, 7),  // "percent"
        QT_MOC_LITERAL(238, 7),  // "message"
        QT_MOC_LITERAL(246, 21),  // "onComputationFinished"
        QT_MOC_LITERAL(268, 18),  // "onComputationError"
        QT_MOC_LITERAL(287, 5)   // "error"
    },
    "DataInputWidget",
    "stationsLoaded",
    "",
    "QList<StationData>",
    "stations",
    "computationRequested",
    "computationComplete",
    "resultText",
    "HypocenterResult",
    "result",
    "onLoadDataFile",
    "onAddRow",
    "onDeleteRow",
    "onComputeClicked",
    "onTableDataChanged",
    "onComputationProgress",
    "percent",
    "message",
    "onComputationFinished",
    "onComputationError",
    "error"
};
#undef QT_MOC_LITERAL
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_DataInputWidget[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   80,    2, 0x06,    1 /* Public */,
       5,    0,   83,    2, 0x06,    3 /* Public */,
       6,    2,   84,    2, 0x06,    4 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
      10,    0,   89,    2, 0x08,    7 /* Private */,
      11,    0,   90,    2, 0x08,    8 /* Private */,
      12,    0,   91,    2, 0x08,    9 /* Private */,
      13,    0,   92,    2, 0x08,   10 /* Private */,
      14,    0,   93,    2, 0x08,   11 /* Private */,
      15,    2,   94,    2, 0x08,   12 /* Private */,
      18,    1,   99,    2, 0x08,   15 /* Private */,
      19,    1,  102,    2, 0x08,   17 /* Private */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString, 0x80000000 | 8,    7,    9,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,   16,   17,
    QMetaType::Void, 0x80000000 | 8,    9,
    QMetaType::Void, QMetaType::QString,   20,

       0        // eod
};

Q_CONSTINIT const QMetaObject DataInputWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_DataInputWidget.offsetsAndSizes,
    qt_meta_data_DataInputWidget,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_DataInputWidget_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<DataInputWidget, std::true_type>,
        // method 'stationsLoaded'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVector<StationData> &, std::false_type>,
        // method 'computationRequested'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'computationComplete'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const HypocenterResult &, std::false_type>,
        // method 'onLoadDataFile'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onAddRow'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onDeleteRow'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onComputeClicked'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onTableDataChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onComputationProgress'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'onComputationFinished'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const HypocenterResult &, std::false_type>,
        // method 'onComputationError'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>
    >,
    nullptr
} };

void DataInputWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DataInputWidget *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->stationsLoaded((*reinterpret_cast< std::add_pointer_t<QList<StationData>>>(_a[1]))); break;
        case 1: _t->computationRequested(); break;
        case 2: _t->computationComplete((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<HypocenterResult>>(_a[2]))); break;
        case 3: _t->onLoadDataFile(); break;
        case 4: _t->onAddRow(); break;
        case 5: _t->onDeleteRow(); break;
        case 6: _t->onComputeClicked(); break;
        case 7: _t->onTableDataChanged(); break;
        case 8: _t->onComputationProgress((*reinterpret_cast< std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 9: _t->onComputationFinished((*reinterpret_cast< std::add_pointer_t<HypocenterResult>>(_a[1]))); break;
        case 10: _t->onComputationError((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (DataInputWidget::*)(const QVector<StationData> & );
            if (_t _q_method = &DataInputWidget::stationsLoaded; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (DataInputWidget::*)();
            if (_t _q_method = &DataInputWidget::computationRequested; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (DataInputWidget::*)(const QString & , const HypocenterResult & );
            if (_t _q_method = &DataInputWidget::computationComplete; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
    }
}

const QMetaObject *DataInputWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DataInputWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DataInputWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int DataInputWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 11)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 11;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 11)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 11;
    }
    return _id;
}

// SIGNAL 0
void DataInputWidget::stationsLoaded(const QVector<StationData> & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void DataInputWidget::computationRequested()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void DataInputWidget::computationComplete(const QString & _t1, const HypocenterResult & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
