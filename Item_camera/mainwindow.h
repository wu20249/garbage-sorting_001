#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QPushButton>
#include <QLabel>
#include <QStackedWidget>
#include "camerawidget.h"
#include "servocontroller.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 新增左侧按钮槽
    void showHomePage();
    void showSmartDetectPage();
    void showManualControlPage();

    // 已有槽函数
    void onRecognizeClicked();
    void onImageCaptured(const QString &imagePath);
    void handleCaptureFailed(const QString &errorMsg);
    void onRecognitionFinished(int exitCode, QProcess::ExitStatus exitStatus);

    void onServo1Rotate90();
    void onServo1Reset();
    void onServo2Rotate90();
    void onServo2Reset();

    //新增多及
    void onOpenClicked();   // 按钮显示"打开"，实际执行"关闭"动作
    void onCloseClicked();  // 按钮显示"关闭"，实际执行"打开"动作


private:
    Ui::MainWindow *ui;

    // 原有成员
    CameraWidget *cameraWidget;
    QProcess *recognitionProcess;
    ServoController *servoController;

    QPushButton *recognizeButton;
    QLabel *resultLabel;
    QLabel *resultImageLabel;

    QPushButton *btnServo1Rotate;
    QPushButton *btnServo1Reset;
    QPushButton *btnServo2Rotate;
    QPushButton *btnServo2Reset;
    QLabel *servoStatusLabel;

    QString lastCapturedImagePath;

    // 新增左侧按钮
    QPushButton *btnHome;
    QPushButton *btnSmartDetect;
    QPushButton *btnManualControl;

    // 堆叠窗口
    QStackedWidget *stackedWidget;

    // 三个页面容器
    QWidget *homePageWidget;
    QWidget *smartDetectWidget;   // 智能检测页面，不含舵机按钮
    QWidget *manualControlWidget; // 手动控制页面，只含舵机按钮

    // 原有功能函数保持不变
    void takePhoto();
    void recognizeImage();
    void parseRecognitionResults(const QString &output);

    // UI组件   新增多及btnHarmfulOpen
    QPushButton *btnHarmfulOpen;   // 声明为成员变量
    QPushButton *btnHarmfulClose;  // 声明为成员变量

    // 舵机控制常量
    const int SERVO_GPIO = 139;       // 20排针11号引脚
    const int PWM_FREQ = 50;          // 50Hz（周期20ms）
    const int MIN_PULSE = 500;        // 0°对应脉冲宽度（μs）
    const int MAX_PULSE = 2500;       // 180°对应脉冲宽度（μs）
    const int OPEN_POSITION = 90;     // 实际打开位置（舵机角度）
    const int CLOSE_POSITION = 0;     // 实际关闭位置（舵机角度）

    // GPIO操作函数
    bool exportGpio(int gpio);
    bool setGpioOut(int gpio);
    bool setGpioValue(int gpio, int value);

    // 舵机控制函数
    void initServo();                 // 初始化GPIO
    void setAngle(int angle);         // 控制舵机转动
};

#endif // MAINWINDOW_H
