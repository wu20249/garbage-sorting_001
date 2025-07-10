#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
//    w.show();
//    w.showFullScreen();
//    w.setFixedSize(1024, 600);
    w.showMaximized();



    return a.exec();
}
