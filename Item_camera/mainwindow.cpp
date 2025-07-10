#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QPixmap>
#include <QDebug>
#include <QFile>
#include <QRegExp>
#include <QMap>
#include <QTimer>
#include <unistd.h>  // 微秒级延时

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , recognitionProcess(new QProcess(this))
    , servoController(new ServoController(this))
{
    ui->setupUi(this);
    setWindowTitle("垃圾分类检测系统");


    // 1. 创建左侧三个导航按钮
    btnHome = new QPushButton("首页");
    btnSmartDetect = new QPushButton("智能检测");
    btnManualControl = new QPushButton("手动控制");

    // 样式
    QString sideButtonStyle = R"(
        QPushButton {
            background-color: #E3F2FD;
            border: none;
            font-size: 20px;
            padding: 20px 0;
        }
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        QPushButton:checked {
            background-color: #42a5f5;
            font-weight: bold;
        }
    )";
    btnHome->setStyleSheet(sideButtonStyle);
    btnSmartDetect->setStyleSheet(sideButtonStyle);
    btnManualControl->setStyleSheet(sideButtonStyle);

    btnHome->setCheckable(true);
    btnSmartDetect->setCheckable(true);
    btnManualControl->setCheckable(true);

    btnHome->setChecked(true);

    // 确保紧贴父容器
    btnHome->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    btnSmartDetect->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    btnManualControl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // 左侧布局
    QVBoxLayout *leftLayout = new QVBoxLayout;
    leftLayout->setSpacing(0);  // 按钮之间无空隙
    leftLayout->setContentsMargins(0, 0, 0, 0); // 与左侧widget无空隙
    leftLayout->addWidget(btnHome);
    leftLayout->addWidget(btnSmartDetect);
    leftLayout->addWidget(btnManualControl);
    leftLayout->addStretch();

    // 左侧区域背景
    QWidget *leftWidget = new QWidget;
    leftWidget->setLayout(leftLayout);
    leftWidget->setFixedWidth(160);
    leftWidget->setStyleSheet(R"(
        "background-color: rgba(255, 255, 255, 200); "
        border-right: 1px solid #cccccc;
    )");

    // 互斥选中
    connect(btnHome, &QPushButton::clicked, this, [this](){
        btnHome->setChecked(true);
        btnSmartDetect->setChecked(false);
        btnManualControl->setChecked(false);
        showHomePage();
    });
    connect(btnSmartDetect, &QPushButton::clicked, this, [this](){
        btnHome->setChecked(false);
        btnSmartDetect->setChecked(true);
        btnManualControl->setChecked(false);
        showSmartDetectPage();
    });
    connect(btnManualControl, &QPushButton::clicked, this, [this](){
        btnHome->setChecked(false);
        btnSmartDetect->setChecked(false);
        btnManualControl->setChecked(true);
        showManualControlPage();
    });


    // 2. 首页页面（简单介绍文本）
    homePageWidget = new QWidget;
    QLabel *homeLabel = new QLabel(
        "<p style='font-size:34px; font-weight:bold; color:#333; margin-bottom:90px;'>欢迎使用垃圾分类检测系统</h2>"
        "<p style='font-size:22px; font-weight:500; color:#555; line-height:1.5;'>"
        "本系统包含两个主要功能：<br>"
        "1. <b>智能检测</b>：利用摄像头和AI技术自动识别垃圾种类，实现快速准确的分类。<br>"
        "2. <b>手动控制</b>：用户通过按钮手动控制垃圾桶开关，适用于用户知道垃圾类别。<br>"
        "请在左侧导航栏选择您需要使用的功能进行操作。"
        "</p>"
    );
    homeLabel->setAlignment(Qt::AlignCenter);
    homeLabel->setWordWrap(true);
    homeLabel->setStyleSheet(R"(
        background-color: rgba(255, 255, 255, 200);
    )");

    QVBoxLayout *homeLayout = new QVBoxLayout(homePageWidget);
    homeLayout->setSpacing(5);
    homeLayout->setContentsMargins(5, 5, 5, 5);
    homeLayout->addWidget(homeLabel);


    // 3. 智能检测页面
    smartDetectWidget = new QWidget;
    QVBoxLayout *smartLayout = new QVBoxLayout(smartDetectWidget);

    // 水平布局（进一步减小间距）
    QHBoxLayout *displayLayout = new QHBoxLayout;
    displayLayout->setSpacing(1); // 间距从1px增至5px（避免过挤），但总宽仍可控
    displayLayout->setContentsMargins(1, 1, 1, 1); // 增加边距避免贴边

    // 计算合理尺寸：右侧可用874px - 间距5px = 869px → 平分后各434px（434*2 +5=873 ≤874）
    int boxWidth = 400;  // 宽度从450px缩减至434px（减少16px）
    int boxHeight = 300; // 高度保持300px（足够显示且不超600px总高）

    // camera
    cameraWidget = new CameraWidget("/dev/video21", this);
    cameraWidget->setFixedSize(boxWidth, boxHeight); // 固定缩减后的尺寸

    // result
    resultImageLabel = new QLabel();
    resultImageLabel->setAlignment(Qt::AlignCenter);
    resultImageLabel->setFixedSize(boxWidth, boxHeight); // 与摄像头尺寸一致
    resultImageLabel->setStyleSheet("background-color: rgba(255,255,255,200);");
    resultImageLabel->setText("");
    // 初始化空图片时使用缩减后的尺寸
    resultImageLabel->setPixmap(QPixmap(boxWidth, boxHeight));

    // 放入垂直布局（简化布局）
    QVBoxLayout *resultBoxLayout = new QVBoxLayout;
    resultBoxLayout->addWidget(resultImageLabel);
    // 移除多余的对齐设置（父布局已控制居中）

    // 添加到水平布局（保持垂直居中）
    displayLayout->addWidget(cameraWidget, 0, Qt::AlignVCenter);
    displayLayout->addLayout(resultBoxLayout);

    // 主布局（保持上下伸缩，居中显示）
    smartLayout->addStretch(1);
    smartLayout->addLayout(displayLayout);
    smartLayout->addStretch(1);


    // 创建结果标签（移至与摄像头同级的位置）
    resultLabel = new QLabel("识别结果类别将显示在这里");
    resultLabel->setAlignment(Qt::AlignCenter);
    resultLabel->setStyleSheet(
        "font-size: 18px; " // 增大字体以便更明显
        "font-weight: bold; "
        "background-color: rgba(255,255,255,200);"
    );


    // 创建识别按钮
    recognizeButton = new QPushButton("识别");
    recognizeButton->setMinimumHeight(60);
    recognizeButton->setMinimumWidth(130);
    recognizeButton->setCursor(Qt::PointingHandCursor);
    recognizeButton->setStyleSheet(R"(
        QPushButton {
            background-color: #4A86E8;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            padding: 4px 8px;
            text-align: center;
            border: 1px solid #dddddd;
        }
    )");

    // 将各组件添加到主布局，并使用伸缩项使其垂直居中
    smartLayout->addStretch(1);       // 顶部伸缩项
    smartLayout->addLayout(displayLayout); // 摄像头和结果框
    smartLayout->addSpacing(-80);
    smartLayout->addWidget(resultLabel);   // 结果标签
    smartLayout->addSpacing(10);
    smartLayout->addWidget(recognizeButton, 0, Qt::AlignCenter); // 按钮居中
    smartLayout->addStretch(1);       // 底部伸缩项


//    // 4. 手动控制页面（整体水平+垂直居中）
//    manualControlWidget = new QWidget;
//    QVBoxLayout *outerLayout = new QVBoxLayout(manualControlWidget); // 最外层垂直布局
//    outerLayout->setContentsMargins(20, 20, 20, 20); // 页面边缘留白

//    // 顶部伸缩项：将内容推到垂直居中位置
//    outerLayout->addStretch();

//    // 创建内容容器（包含标签和按钮组）
//    QWidget *contentWidget = new QWidget;
//    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget); // 内容容器的垂直布局
//    contentLayout->setSpacing(80); // 标签与按钮组的间距

//    // 1. 状态标签（水平居中）
//    servoStatusLabel = new QLabel("就绪");
//    servoStatusLabel->setStyleSheet(R"(
//        color: #27AE60;
//        font-size: 24px;
//        font-weight: bold;
//        background-color: #E9F5F0;
//        border-radius: 5px;
//        padding: 30px 20px;
//        min-width: 100px;
//    )");
//    servoStatusLabel->setAlignment(Qt::AlignCenter); // 水平居中
//    contentLayout->addWidget(servoStatusLabel);

//    // 2. 按钮组容器（使用网格布局实现两行两列）
//    QWidget *buttonsContainer = new QWidget;
//    QGridLayout *buttonsLayout = new QGridLayout(buttonsContainer);
//    buttonsLayout->setSpacing(25); // 按钮之间的水平和垂直间距
//    buttonsLayout->setContentsMargins(0, 0, 0, 0);

//    // 按钮1：可回收垃圾桶：开盖（第一行第一列）
//    btnServo1Rotate = new QPushButton("可回收垃圾桶：开盖");
//    btnServo1Rotate->setMinimumHeight(60);
//    btnServo1Rotate->setCursor(Qt::PointingHandCursor);
//    btnServo1Rotate->setStyleSheet(R"(
//        QPushButton {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #37b24d, stop:1 #2b8a3e);
//            color: white;
//            border-radius: 6px;
//            font-size: 16px;
//            padding: 8px 15px;
//            min-width: 180px; /* 增加最小宽度，确保按钮文字不换行 */
//        }
//        QPushButton:hover {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #40c057, stop:1 #34a34c);
//        }
//        QPushButton:pressed {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #2f9e44, stop:1 #217034);
//        }
//    )");

//    // 按钮2：可回收垃圾桶：关盖（第二行第一列）
//    btnServo1Reset = new QPushButton("可回收垃圾桶：关盖");
//    btnServo1Reset->setMinimumHeight(60);
//    btnServo1Reset->setCursor(Qt::PointingHandCursor);
//    btnServo1Reset->setStyleSheet(R"(
//        QPushButton {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f03e3e, stop:1 #c92a2a);
//            color: white;
//            border-radius: 6px;
//            font-size: 16px;
//            padding: 8px 15px;
//            min-width: 180px;
//        }
//        QPushButton:hover {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #fa5252, stop:1 #e03131);
//        }
//        QPushButton:pressed {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #e03131, stop:1 #a61e1e);
//        }
//    )");

//    // 按钮3：其它垃圾桶：开盖（第一行第二列）
//    btnServo2Rotate = new QPushButton("其它垃圾桶：开盖");
//    btnServo2Rotate->setMinimumHeight(60);
//    btnServo2Rotate->setCursor(Qt::PointingHandCursor);
//    btnServo2Rotate->setStyleSheet(R"(
//        QPushButton {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #4c6ef5, stop:1 #3b5bdb);
//            color: white;
//            border-radius: 6px;
//            font-size: 16px;
//            padding: 8px 15px;
//            min-width: 180px;
//        }
//        QPushButton:hover {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #5c7cfa, stop:1 #4263eb);
//        }
//        QPushButton:pressed {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #4263eb, stop:1 #2c4ade);
//        }
//    )");

//    // 按钮4：其它垃圾桶：关盖（第二行第二列）
//    btnServo2Reset = new QPushButton("其它垃圾桶：关盖");
//    btnServo2Reset->setMinimumHeight(60);
//    btnServo2Reset->setCursor(Qt::PointingHandCursor);
//    btnServo2Reset->setStyleSheet(R"(
//        QPushButton {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f76707, stop:1 #e65c00);
//            color: white;
//            border-radius: 6px;
//            font-size: 16px;
//            padding: 8px 15px;
//            min-width: 180px;
//        }
//        QPushButton:hover {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #fd7e14, stop:1 #f06543);
//        }
//        QPushButton:pressed {
//            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #e8590c, stop:1 #d9480f);
//        }
//    )");

//    // 按网格布局排列按钮（两行两列）
//    // 第一行：可回收开盖（0,0）、其它开盖（0,1）
//    // 第二行：可回收关盖（1,0）、其它关盖（1,1）
//    buttonsLayout->addWidget(btnServo1Rotate, 0, 0);  // 第一行第一列
//    buttonsLayout->addWidget(btnServo2Rotate, 0, 1);  // 第一行第二列
//    buttonsLayout->addWidget(btnServo1Reset, 1, 0);   // 第二行第一列
//    buttonsLayout->addWidget(btnServo2Reset, 1, 1);   // 第二行第二列

//    // 设置列宽比例，确保两列按钮宽度一致
//    buttonsLayout->setColumnStretch(0, 1);
//    buttonsLayout->setColumnStretch(1, 1);

//    // 添加按钮组容器到内容布局（水平居中）
//    contentLayout->addWidget(buttonsContainer, 0, Qt::AlignCenter);

//    // 将内容容器添加到外层布局
//    outerLayout->addWidget(contentWidget, 0, Qt::AlignCenter); // 内容容器水平+垂直居中

//    // 底部伸缩项：与顶部平衡，保持垂直居中
//    outerLayout->addStretch();
    // 4. 手动控制页面（整体水平+垂直居中）
    manualControlWidget = new QWidget;
    QVBoxLayout *outerLayout = new QVBoxLayout(manualControlWidget); // 最外层垂直布局
    outerLayout->setContentsMargins(20, 20, 20, 20); // 页面边缘留白

    // 顶部伸缩项：将内容推到垂直居中位置
    outerLayout->addStretch();

    // 创建内容容器（包含标签和按钮组）
    QWidget *contentWidget = new QWidget;
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget); // 内容容器的垂直布局
    contentLayout->setSpacing(80); // 标签与按钮组的间距

    // 1. 状态标签（水平居中）
    servoStatusLabel = new QLabel("就绪");
    servoStatusLabel->setStyleSheet(R"(
        color: #27AE60;
        font-size: 24px;
        font-weight: bold;
        background-color: #E9F5F0;
        border-radius: 5px;
        padding: 30px 20px;
        min-width: 100px;
    )");
    servoStatusLabel->setAlignment(Qt::AlignCenter); // 水平居中
    contentLayout->addWidget(servoStatusLabel);

    // 2. 按钮组容器（使用网格布局实现两行三列）
    QWidget *buttonsContainer = new QWidget;
    QGridLayout *buttonsLayout = new QGridLayout(buttonsContainer);
    buttonsLayout->setSpacing(25); // 按钮之间的水平和垂直间距
    buttonsLayout->setContentsMargins(0, 0, 0, 0);

    // ---------------------- 可回收垃圾桶按钮 ----------------------
    // 可回收垃圾桶：开盖（第一行第一列）
    btnServo1Rotate = new QPushButton("可回收垃圾桶：开盖");
    btnServo1Rotate->setMinimumHeight(60);
    btnServo1Rotate->setCursor(Qt::PointingHandCursor);
    btnServo1Rotate->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #37b24d, stop:1 #2b8a3e);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #40c057, stop:1 #34a34c);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #2f9e44, stop:1 #217034);
        }
    )");

    // 可回收垃圾桶：关盖（第二行第一列）
    btnServo1Reset = new QPushButton("可回收垃圾桶：关盖");
    btnServo1Reset->setMinimumHeight(60);
    btnServo1Reset->setCursor(Qt::PointingHandCursor);
    btnServo1Reset->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f03e3e, stop:1 #c92a2a);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #fa5252, stop:1 #e03131);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #e03131, stop:1 #a61e1e);
        }
    )");

    // ---------------------- 其它垃圾桶按钮 ----------------------
    // 其它垃圾桶：开盖（第一行第二列）
    btnServo2Rotate = new QPushButton("其它垃圾桶：开盖");
    btnServo2Rotate->setMinimumHeight(60);
    btnServo2Rotate->setCursor(Qt::PointingHandCursor);
    btnServo2Rotate->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #4c6ef5, stop:1 #3b5bdb);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #5c7cfa, stop:1 #4263eb);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #4263eb, stop:1 #2c4ade);
        }
    )");

    // 其它垃圾桶：关盖（第二行第二列）
    btnServo2Reset = new QPushButton("其它垃圾桶：关盖");
    btnServo2Reset->setMinimumHeight(60);
    btnServo2Reset->setCursor(Qt::PointingHandCursor);
    btnServo2Reset->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f76707, stop:1 #e65c00);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #fd7e14, stop:1 #f06543);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #e8590c, stop:1 #d9480f);
        }
    )");

    // ---------------------- 有害垃圾桶按钮（新增） ----------------------
    // 有害垃圾桶：开盖（第一行第三列）
    btnHarmfulOpen = new QPushButton("有害垃圾桶：开盖");
    btnHarmfulOpen->setMinimumHeight(60);
    btnHarmfulOpen->setCursor(Qt::PointingHandCursor);
    btnHarmfulOpen->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #9775fa, stop:1 #7950f2);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #a78bfa, stop:1 #8b5cf6);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #8b5cf6, stop:1 #6d28d9);
        }
    )");

    // 有害垃圾桶：关盖（第二行第三列）
    btnHarmfulClose = new QPushButton("有害垃圾桶：关盖");
    btnHarmfulClose->setMinimumHeight(60);
    btnHarmfulClose->setCursor(Qt::PointingHandCursor);
    btnHarmfulClose->setStyleSheet(R"(
        QPushButton {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f43f5e, stop:1 #ec4899);
            color: white;
            border-radius: 6px;
            font-size: 16px;
            padding: 8px 15px;
            min-width: 180px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #fb7185, stop:1 #f472b6);
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f472b6, stop:1 #be185d);
        }
    )");

    // 按网格布局排列所有按钮（两行三列）
    buttonsLayout->addWidget(btnServo1Rotate, 0, 0);    // 第一行第一列
    buttonsLayout->addWidget(btnServo2Rotate, 0, 1);    // 第一行第二列
    buttonsLayout->addWidget(btnHarmfulOpen, 0, 2);     // 第一行第三列（新增）
    buttonsLayout->addWidget(btnServo1Reset, 1, 0);     // 第二行第一列
    buttonsLayout->addWidget(btnServo2Reset, 1, 1);     // 第二行第二列
    buttonsLayout->addWidget(btnHarmfulClose, 1, 2);    // 第二行第三列（新增）

    // 设置列宽比例，确保三列按钮宽度一致
    buttonsLayout->setColumnStretch(0, 1);
    buttonsLayout->setColumnStretch(1, 1);
    buttonsLayout->setColumnStretch(2, 1);  // 新增第三列的伸缩比例

    // 将按钮容器添加到内容布局（水平居中）
    contentLayout->addWidget(buttonsContainer, 0, Qt::AlignCenter);

    // 将内容容器添加到外层布局
    outerLayout->addWidget(contentWidget, 0, Qt::AlignCenter); // 内容容器水平+垂直居中

    // 底部伸缩项：与顶部平衡，保持垂直居中
    outerLayout->addStretch();

    // 有害垃圾桶（使用你已有的函数）
    connect(btnHarmfulOpen, &QPushButton::clicked, this, &MainWindow::onOpenClicked);
    connect(btnHarmfulClose, &QPushButton::clicked, this, &MainWindow::onCloseClicked);
    initServo();

    // 5. 堆叠控件，右侧显示对应页面
    stackedWidget = new QStackedWidget;
    stackedWidget->addWidget(homePageWidget);
    stackedWidget->addWidget(smartDetectWidget);
    stackedWidget->addWidget(manualControlWidget);

    // 6. 主布局横向放置左侧导航和右侧内容
    QHBoxLayout *mainLayout = new QHBoxLayout;
    mainLayout->addWidget(leftWidget);
    mainLayout->addWidget(stackedWidget, 1);

    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);

    // 7. 连接左侧按钮切换页面槽
    connect(btnHome, &QPushButton::clicked, this, &MainWindow::showHomePage);
    connect(btnSmartDetect, &QPushButton::clicked, this, &MainWindow::showSmartDetectPage);
    connect(btnManualControl, &QPushButton::clicked, this, &MainWindow::showManualControlPage);

    // 8. 连接智能检测相关信号槽（保持你原来的代码）
    connect(cameraWidget, &CameraWidget::captureFailed, this, &MainWindow::handleCaptureFailed);
    connect(recognizeButton, &QPushButton::clicked, this, &MainWindow::onRecognizeClicked);
    connect(cameraWidget, &CameraWidget::imageSaved, this, &MainWindow::onImageCaptured);
    connect(recognitionProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onRecognitionFinished);

    // 9. 连接手动控制按钮槽
    connect(btnServo1Rotate, &QPushButton::clicked, this, &MainWindow::onServo1Rotate90);
    connect(btnServo1Reset, &QPushButton::clicked, this, &MainWindow::onServo1Reset);
    connect(btnServo2Rotate, &QPushButton::clicked, this, &MainWindow::onServo2Rotate90);
    connect(btnServo2Reset, &QPushButton::clicked, this, &MainWindow::onServo2Reset);
    connect(servoController, &ServoController::statusMessage, servoStatusLabel, &QLabel::setText);

    // 10. 启动舵机线程和摄像头（保持你原代码）
    servoController->start();
    cameraWidget->startCamera();

    // 11. 默认显示首页
    stackedWidget->setCurrentWidget(homePageWidget);

    // 12. 设置整体样式（保留你原样式） /home/elf/QT_Item/Item_camera/
    //this->setStyleSheet(R"(QMainWindow { background-color: #E9EFF6; })");
    this->setStyleSheet(R"(
        QMainWindow {
            /* 替换为你的图片路径（绝对路径或资源路径）/mnt/sdcard  */
            border-image: url(/mnt/sdcard/bg4.png) 0;
        }
    )");

}

// 槽函数实现
void MainWindow::showHomePage()
{
    stackedWidget->setCurrentWidget(homePageWidget);
    // 进入首页可以暂停摄像头等，根据需求自行添加
}

void MainWindow::showSmartDetectPage()
{
    stackedWidget->setCurrentWidget(smartDetectWidget);
    cameraWidget->startCamera();
}

void MainWindow::showManualControlPage()
{
    stackedWidget->setCurrentWidget(manualControlWidget);
    cameraWidget->stopCamera();
}


//
void MainWindow::onRecognizeClicked()
{
    recognizeButton->setEnabled(false);  // 防止多次点击
    cameraWidget->captureImage();        // 点击识别按钮时先拍照
}

void MainWindow::onImageCaptured(const QString &imagePath)
{
    lastCapturedImagePath = imagePath;
    recognizeImage();                    // 拍照完成后立即调用识别函数
}

void MainWindow::handleCaptureFailed(const QString &errorMsg)
{
    recognizeButton->setEnabled(true);
    resultLabel->setText("拍照失败：" + errorMsg);
}

MainWindow::~MainWindow()
{
    // 停止舵机控制线程
    servoController->stop();

    delete ui;
}

// 拍照功能
void MainWindow::takePhoto()
{
    cameraWidget->captureImage();
}

// 显示拍摄的图像


// 执行图像识别
void MainWindow::recognizeImage()
{
    // 检查图像路径是否有效
    if (lastCapturedImagePath.isEmpty() || !QFile::exists(lastCapturedImagePath)) {
        qDebug() << "错误：图像路径无效 - " << lastCapturedImagePath;
        resultLabel->setText("图像路径无效");
        return;
    }

    // 设置识别程序和参数
    QString program = "/home/elf/rknn_yolov8_demo04/rknn_yolov8_demo";
    QString modelPath = "/home/elf/rknn_yolov8_demo04/model/rubbish.rknn";
    QString inputPath = lastCapturedImagePath;

    // 检查识别程序是否存在
    if (!QFile::exists(program)) {
        qDebug() << "错误：识别程序不存在 - " << program;
        resultLabel->setText("识别程序不存在");
        return;
    }

    // 检查模型文件是否存在
    if (!QFile::exists(modelPath)) {
        qDebug() << "错误：模型文件不存在 - " << modelPath;
        resultLabel->setText("模型文件不存在");
        return;
    }

    // 设置识别进程参数
    QStringList arguments;
    arguments << modelPath << inputPath;

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    recognitionProcess->setProcessEnvironment(env);
    recognitionProcess->setWorkingDirectory("/home/elf/rknn_yolov8_demo04");

    // 启动识别进程
    qDebug() << "启动识别进程:" << program << arguments;
    recognitionProcess->start(program, arguments);

    // 等待进程启动和完成
    if (!recognitionProcess->waitForStarted(5000)) {
        qDebug() << "错误：无法启动识别进程!";
        resultLabel->setText("无法启动识别进程");
        return;
    }

    if (!recognitionProcess->waitForFinished(30000)) {
        qDebug() << "错误：识别进程执行超时!";
        resultLabel->setText("识别超时，请重试");
        return;
    }

    // 读取识别程序输出
    QString stdout = recognitionProcess->readAllStandardOutput();
    QString stderr = recognitionProcess->readAllStandardError();

    qDebug() << "识别程序标准输出:\n" << stdout;
    qDebug() << "识别程序错误输出:\n" << stderr;

    // 解析识别结果
    parseRecognitionResults(stdout);
}


void MainWindow::parseRecognitionResults(const QString &output)
{
    // 清空之前的结果
    resultLabel->clear();

    // 细化垃圾分类映射：新增有害垃圾类别
    QMap<QString, QString> wasteClassification = {
        // 可回收物
        {"disposable", "可回收物"},
        {"book", "可回收物"},
        {"charger", "可回收物"},
        {"bag", "可回收物"},
        {"pccookware", "可回收物"},
        {"pcware", "可回收物"},
        {"pctoy", "可回收物"},
        {"pchanger", "可回收物"},
        {"deliverypaper", "可回收物"},
        {"wire", "可回收物"},
        {"oldcloth", "可回收物"},
        {"can", "可回收物"},
        {"glassware", "可回收物"},
        {"carton", "可回收物"},
        {"paperbag", "可回收物"},
        {"flowerpot", "可回收物"},
        {"spice", "可回收物"},
        {"winebottle", "可回收物"},
        {"metalcooktool", "可回收物"},
        {"metalware", "可回收物"},
        {"metalcan", "可回收物"},
        {"pan", "可回收物"},
        {"oilbox", "可回收物"},
        {"drinkbottle", "可回收物"},
        {"drinkbox", "可回收物"},

        // 其他垃圾（保留无害的物品）
        {"leftover", "其他垃圾"},
        {"bin", "其他垃圾"},
        {"bigbone", "其他垃圾"},
        {"pillow", "其他垃圾"},
        {"fruitpeel", "其他垃圾"},
        {"towel", "其他垃圾"},
        {"plushdoll", "其他垃圾"},
        {"dirtyplastic", "其他垃圾"},
        {"dirtypaper", "其他垃圾"},
        {"care", "其他垃圾"},
        {"butt", "其他垃圾"},
        {"pick", "其他垃圾"},
        {"board", "其他垃圾"},
        {"chopstick", "其他垃圾"},
        {"tealeaf", "其他垃圾"},
        {"vegleaf", "其他垃圾"},
        {"eggshell", "其他垃圾"},
        {"ceramic", "其他垃圾"},
        {"shoe", "其他垃圾"},
        {"fishbone", "其他垃圾"},

        // 新增：有害垃圾（从原来的其他垃圾中分离）
        {"battery", "有害垃圾"},
        {"ointment", "有害垃圾"},
        {"drug", "有害垃圾"}
    };

    // 定义正则表达式匹配识别结果行 (标签 @ (坐标) 置信度)
    QRegExp regExp("(\\S+) @ \\(\\d+ \\d+ \\d+ \\d+\\) (\\d+\\.\\d+)");
    int index = 0;
    QString labelText;
    bool hasRecyclable = false;  // 标记是否有可回收物
    bool hasOtherWaste = false;  // 标记是否有其它垃圾
    bool hasHazardous = false;   // 新增：标记是否有有害垃圾

    while ((index = regExp.indexIn(output, index)) != -1) {
        QString label = regExp.cap(1);       // 识别标签
        QString confidenceStr = regExp.cap(2); // 置信度
        float confidence = confidenceStr.toFloat();

        // 置信度过滤（阈值0.3）
        if (confidence > 0.3f) {
            qDebug() << "捕获到类别标签:" << label;
            qDebug() << "置信度:" << confidence;

            // 构建标签文本（显示类别和垃圾分类）
            if (wasteClassification.contains(label)) {
                QString classification = wasteClassification[label];
                labelText += label + " (" + classification + ")";

                // 判断分类类型
                if (classification == "可回收物") {
                    hasRecyclable = true;
                } else if (classification == "其他垃圾") {
                    hasOtherWaste = true;
                } else if (classification == "有害垃圾") {
                    hasHazardous = true;  // 新增：有害垃圾标记
                }
            } else {
                // 未分类的统一归为其它垃圾
                labelText += label + " (其它垃圾)";
                hasOtherWaste = true;
            }
        }

        index += regExp.matchedLength();
    }

    // 在UI上显示识别结果
    if (!labelText.isEmpty()) {
        resultLabel->setText("识别结果:" + labelText);

        // 如果有可回收物，打开可回收物垃圾桶
        if (hasRecyclable) {
            servoController->rotateServo1To90();
            servoStatusLabel->setText("已打开可回收垃圾桶");

            // 延迟7秒后关闭可回收垃圾桶
            QTimer::singleShot(7000, this, &MainWindow::onServo1Reset);
        }

        // 如果有其它垃圾，打开其它垃圾桶
        if (hasOtherWaste) {
            servoController->rotateServo2To90();
            servoStatusLabel->setText("已打开其它垃圾桶");

            // 延迟7秒后关闭其它垃圾桶
            QTimer::singleShot(7000, this, &MainWindow::onServo2Reset);
        }

        // 新增：如果有有害垃圾，打开有害垃圾桶
        if (hasHazardous) {
            // 调用有害垃圾桶的打开函数（使用你已有的onOpenClicked）
            onOpenClicked();  // 打开有害垃圾桶

            // 延迟7秒后关闭有害垃圾桶
//            QTimer::singleShot(7000, this, &MainWindow::onCloseClicked);
            // 延迟7秒后关闭垃圾桶（直接旋转到110°）
            QTimer::singleShot(7000, this, [=]() {
                setAngle(110);  // 直接调用setAngle，跳过onCloseClicked
                servoStatusLabel->setText("有害垃圾桶：关盖完成");
            });
        }
    } else {
        resultLabel->setText("未识别到有效垃圾");
    }
}



// 处理识别完成
void MainWindow::onRecognitionFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    recognizeButton->setEnabled(true);

    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        // 识别成功，显示结果图片
        QString resultPath = "/home/elf/rknn_yolov8_demo04/out.png";

        QFile resultFile(resultPath);
        if (resultFile.exists()) {
            qDebug() << "识别结果文件存在:" << resultPath;

            QPixmap resultPixmap(resultPath);
            if (!resultPixmap.isNull()) {
                resultImageLabel->setPixmap(resultPixmap.scaled(
                    resultImageLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation
                ));
            } else {
                qDebug() << "无法加载图片，但文件存在。可能格式错误或文件损坏。";
                resultImageLabel->setText("结果图片格式错误");
            }
        } else {
            qDebug() << "识别结果文件不存在:" << resultPath;
            resultImageLabel->setText("未找到识别结果文件");
        }
    } else {
        qDebug() << "识别失败，退出代码:" << exitCode;
        resultImageLabel->setText("识别失败，请重试");
    }
}

// 舵机控制槽函数
void MainWindow::onServo1Rotate90()
{
    if (servoController) {
        servoController->rotateServo1To90();
    }
}

void MainWindow::onServo1Reset()
{
    if (servoController) {
        servoController->resetServo1();
    }
}

void MainWindow::onServo2Rotate90()
{
    if (servoController) {
        servoController->rotateServo2To90();
    }
}

void MainWindow::onServo2Reset()
{
    if (servoController) {
        servoController->resetServo2();
    }
}

//新增多及
// 点击"打开"按钮
void MainWindow::onOpenClicked() {
    servoStatusLabel->setText("有害垃圾桶：开盖完成");
    setAngle(CLOSE_POSITION);  // 旋转到0°
}

// 点击"关闭"按钮
void MainWindow::onCloseClicked() {
    servoStatusLabel->setText("有害垃圾桶：关盖完成");
    setAngle(OPEN_POSITION);  // 旋转到90°
}

// 导出GPIO并设置为输出模式
bool MainWindow::exportGpio(int gpio) {
    QFile exportFile("/sys/class/gpio/export");
    if (!exportFile.open(QIODevice::WriteOnly)) {
        qDebug() << "GPIO导出失败";
        return false;
    }
    exportFile.write(QString::number(gpio).toUtf8());
    exportFile.close();
    usleep(100000);  // 等待100ms
    return true;
}

// 设置GPIO为输出模式
bool MainWindow::setGpioOut(int gpio) {
    QFile dirFile(QString("/sys/class/gpio/gpio%1/direction").arg(gpio));
    if (!dirFile.open(QIODevice::WriteOnly)) {
        qDebug() << "GPIO方向设置失败";
        return false;
    }
    dirFile.write("out");
    dirFile.close();
    return true;
}

// 设置GPIO电平
bool MainWindow::setGpioValue(int gpio, int value) {
    QFile valueFile(QString("/sys/class/gpio/gpio%1/value").arg(gpio));
    if (!valueFile.open(QIODevice::WriteOnly)) {
        qDebug() << "GPIO电平设置失败";
        return false;
    }
    valueFile.write(QString::number(value).toUtf8());
    valueFile.close();
    return true;
}

// 初始化舵机（保持当前位置）
void MainWindow::initServo() {
    if (exportGpio(SERVO_GPIO) && setGpioOut(SERVO_GPIO)) {
        setGpioValue(SERVO_GPIO, 0);  // 初始置低电平
        qDebug() << "舵机初始化完成（保持当前位置）";
    }
}

// 软件模拟PWM控制角度
void MainWindow::setAngle(int angle) {
    // 限制角度范围
    angle = qBound(0, angle, 180);

    // 计算脉冲宽度
    int pulseWidth = MIN_PULSE + (angle * (MAX_PULSE - MIN_PULSE)) / 180;
    int period = 20000;  // 20ms周期

    // 转换为无符号类型（修复警告）
    unsigned int highTime = static_cast<unsigned int>(pulseWidth);
    unsigned int lowTime = static_cast<unsigned int>(period - pulseWidth);

    // 发送50个PWM周期
    for (int i = 0; i < 50; ++i) {
        setGpioValue(SERVO_GPIO, 1);  // 高电平
        usleep(highTime);

        setGpioValue(SERVO_GPIO, 0);  // 低电平
        usleep(lowTime);
    }
}
