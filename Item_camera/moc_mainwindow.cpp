/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.10)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.10. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[20];
    char stringdata0[295];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 12), // "showHomePage"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 19), // "showSmartDetectPage"
QT_MOC_LITERAL(4, 45, 21), // "showManualControlPage"
QT_MOC_LITERAL(5, 67, 18), // "onRecognizeClicked"
QT_MOC_LITERAL(6, 86, 15), // "onImageCaptured"
QT_MOC_LITERAL(7, 102, 9), // "imagePath"
QT_MOC_LITERAL(8, 112, 19), // "handleCaptureFailed"
QT_MOC_LITERAL(9, 132, 8), // "errorMsg"
QT_MOC_LITERAL(10, 141, 21), // "onRecognitionFinished"
QT_MOC_LITERAL(11, 163, 8), // "exitCode"
QT_MOC_LITERAL(12, 172, 20), // "QProcess::ExitStatus"
QT_MOC_LITERAL(13, 193, 10), // "exitStatus"
QT_MOC_LITERAL(14, 204, 16), // "onServo1Rotate90"
QT_MOC_LITERAL(15, 221, 13), // "onServo1Reset"
QT_MOC_LITERAL(16, 235, 16), // "onServo2Rotate90"
QT_MOC_LITERAL(17, 252, 13), // "onServo2Reset"
QT_MOC_LITERAL(18, 266, 13), // "onOpenClicked"
QT_MOC_LITERAL(19, 280, 14) // "onCloseClicked"

    },
    "MainWindow\0showHomePage\0\0showSmartDetectPage\0"
    "showManualControlPage\0onRecognizeClicked\0"
    "onImageCaptured\0imagePath\0handleCaptureFailed\0"
    "errorMsg\0onRecognitionFinished\0exitCode\0"
    "QProcess::ExitStatus\0exitStatus\0"
    "onServo1Rotate90\0onServo1Reset\0"
    "onServo2Rotate90\0onServo2Reset\0"
    "onOpenClicked\0onCloseClicked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   79,    2, 0x08 /* Private */,
       3,    0,   80,    2, 0x08 /* Private */,
       4,    0,   81,    2, 0x08 /* Private */,
       5,    0,   82,    2, 0x08 /* Private */,
       6,    1,   83,    2, 0x08 /* Private */,
       8,    1,   86,    2, 0x08 /* Private */,
      10,    2,   89,    2, 0x08 /* Private */,
      14,    0,   94,    2, 0x08 /* Private */,
      15,    0,   95,    2, 0x08 /* Private */,
      16,    0,   96,    2, 0x08 /* Private */,
      17,    0,   97,    2, 0x08 /* Private */,
      18,    0,   98,    2, 0x08 /* Private */,
      19,    0,   99,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString,    9,
    QMetaType::Void, QMetaType::Int, 0x80000000 | 12,   11,   13,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->showHomePage(); break;
        case 1: _t->showSmartDetectPage(); break;
        case 2: _t->showManualControlPage(); break;
        case 3: _t->onRecognizeClicked(); break;
        case 4: _t->onImageCaptured((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 5: _t->handleCaptureFailed((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 6: _t->onRecognitionFinished((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QProcess::ExitStatus(*)>(_a[2]))); break;
        case 7: _t->onServo1Rotate90(); break;
        case 8: _t->onServo1Reset(); break;
        case 9: _t->onServo2Rotate90(); break;
        case 10: _t->onServo2Reset(); break;
        case 11: _t->onOpenClicked(); break;
        case 12: _t->onCloseClicked(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.data,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
