/****************************************************************************
** Meta object code from reading C++ file 'servocontroller.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.10)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "servocontroller.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'servocontroller.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.10. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ServoController_t {
    QByteArrayData data[13];
    char stringdata0[201];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ServoController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ServoController_t qt_meta_stringdata_ServoController = {
    {
QT_MOC_LITERAL(0, 0, 15), // "ServoController"
QT_MOC_LITERAL(1, 16, 13), // "statusMessage"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 7), // "message"
QT_MOC_LITERAL(4, 39, 16), // "commandCompleted"
QT_MOC_LITERAL(5, 56, 16), // "rotateServo1To90"
QT_MOC_LITERAL(6, 73, 11), // "resetServo1"
QT_MOC_LITERAL(7, 85, 16), // "rotateServo2To90"
QT_MOC_LITERAL(8, 102, 11), // "resetServo2"
QT_MOC_LITERAL(9, 114, 22), // "executeRotate90Command"
QT_MOC_LITERAL(10, 137, 20), // "executeReset1Command"
QT_MOC_LITERAL(11, 158, 21), // "executeRotate2Command"
QT_MOC_LITERAL(12, 180, 20) // "executeReset2Command"

    },
    "ServoController\0statusMessage\0\0message\0"
    "commandCompleted\0rotateServo1To90\0"
    "resetServo1\0rotateServo2To90\0resetServo2\0"
    "executeRotate90Command\0executeReset1Command\0"
    "executeRotate2Command\0executeReset2Command"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ServoController[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   64,    2, 0x06 /* Public */,
       4,    0,   67,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   68,    2, 0x0a /* Public */,
       6,    0,   69,    2, 0x0a /* Public */,
       7,    0,   70,    2, 0x0a /* Public */,
       8,    0,   71,    2, 0x0a /* Public */,
       9,    0,   72,    2, 0x08 /* Private */,
      10,    0,   73,    2, 0x08 /* Private */,
      11,    0,   74,    2, 0x08 /* Private */,
      12,    0,   75,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void ServoController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ServoController *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->statusMessage((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->commandCompleted(); break;
        case 2: _t->rotateServo1To90(); break;
        case 3: _t->resetServo1(); break;
        case 4: _t->rotateServo2To90(); break;
        case 5: _t->resetServo2(); break;
        case 6: _t->executeRotate90Command(); break;
        case 7: _t->executeReset1Command(); break;
        case 8: _t->executeRotate2Command(); break;
        case 9: _t->executeReset2Command(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ServoController::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ServoController::statusMessage)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ServoController::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ServoController::commandCompleted)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ServoController::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_ServoController.data,
    qt_meta_data_ServoController,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ServoController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ServoController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ServoController.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int ServoController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void ServoController::statusMessage(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ServoController::commandCompleted()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
