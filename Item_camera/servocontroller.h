#ifndef SERVOCONTROLLER_H
#define SERVOCONTROLLER_H

#include <QObject>
#include <QString>
#include <QThread>
#include <QMutex>

class ServoController : public QObject
{
    Q_OBJECT

public:
    explicit ServoController(QObject *parent = nullptr);
    ~ServoController();

    // 启动和停止控制线程
    void start();
    void stop();

public slots:
    // 线程安全的命令接口
    void rotateServo1To90();
    void resetServo1();
    void rotateServo2To90();
    void resetServo2();

signals:
    void statusMessage(const QString &message);
    void commandCompleted();

private slots:
    // 在线程中执行的实际命令
    void executeRotate90Command();
    void executeReset1Command();
    void executeRotate2Command();
    void executeReset2Command();

private:
    // 舵机路径定义
    const QString basePath1 = "/sys/class/pwm/pwmchip0";
    const QString basePath2 = "/sys/class/pwm/pwmchip1";

    // 线程和同步
    QThread *workerThread;
    QMutex commandMutex;
    bool threadRunning;

    // 辅助函数
    bool writeValue(const QString &chipPath, const QString &fileName, const QString &value);
    bool exportPwmChannel(const QString &chipPath, int channel);
    bool unexportPwmChannel(const QString &chipPath, int channel);
    bool isPwmChannelExported(const QString &chipPath, int channel);
};

#endif // SERVOCONTROLLER_H
