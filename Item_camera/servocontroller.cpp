#include "servocontroller.h"
#include <QFile>

ServoController::ServoController(QObject *parent)
    : QObject(parent)
    , workerThread(new QThread(this))
    , threadRunning(false)
{
    this->moveToThread(workerThread);

    connect(workerThread, &QThread::started, [this]() {
        threadRunning = true;
        emit statusMessage("垃圾桶准备就绪");
    });

    connect(workerThread, &QThread::finished, [this]() {
        threadRunning = false;
        emit statusMessage("舵机控制线程已停止");
    });
}

ServoController::~ServoController()
{
    stop();
}

void ServoController::start()
{
    if (!workerThread->isRunning()) {
        workerThread->start();
        QMetaObject::invokeMethod(this, [this]() {
            exportPwmChannel(basePath2, 0);
        }, Qt::QueuedConnection);
    }
}

void ServoController::stop()
{
    if (workerThread->isRunning()) {
        workerThread->quit();
        workerThread->wait();
    }
}

void ServoController::rotateServo1To90()
{
    if (threadRunning) {
        QMetaObject::invokeMethod(this, &ServoController::executeRotate90Command, Qt::QueuedConnection);
    }
}

void ServoController::resetServo1()
{
    if (threadRunning) {
        QMetaObject::invokeMethod(this, &ServoController::executeReset1Command, Qt::QueuedConnection);
    }
}

void ServoController::rotateServo2To90()
{
    if (threadRunning) {
        QMetaObject::invokeMethod(this, &ServoController::executeRotate2Command, Qt::QueuedConnection);
    }
}

void ServoController::resetServo2()
{
    if (threadRunning) {
        QMetaObject::invokeMethod(this, &ServoController::executeReset2Command, Qt::QueuedConnection);
    }
}

void ServoController::executeRotate90Command()
{
    QMutexLocker locker(&commandMutex);
    emit statusMessage("可回收垃圾桶：开盖");

    QString path = basePath1 + "/pwm0";

    if (!exportPwmChannel(basePath1, 0)) {
        emit statusMessage("可回收垃圾桶：开盖失败");
        emit commandCompleted();
        return;
    }

    if (!writeValue(path, "period", "20000000")) goto cleanup;
    if (!writeValue(path, "duty_cycle", "800000")) goto cleanup;
    if (!writeValue(path, "polarity", "normal")) goto cleanup;
    if (!writeValue(path, "enable", "1")) goto cleanup;

    emit statusMessage("可回收垃圾桶：开盖完成");
    emit commandCompleted();
    return;

cleanup:
    unexportPwmChannel(basePath1, 0);
    emit statusMessage("可回收垃圾桶：开盖操作失败");
    emit commandCompleted();
}

void ServoController::executeReset1Command()
{
    QMutexLocker locker(&commandMutex);
    emit statusMessage("可回收垃圾桶：关盖");

    QString path = basePath1 + "/pwm0";

    if (!exportPwmChannel(basePath1, 0)) {
        emit statusMessage("可回收垃圾桶：关盖失败");
        emit commandCompleted();
        return;
    }

    if (!writeValue(path, "period", "20000000")) goto cleanup;
    if (!writeValue(path, "duty_cycle", "1800000")) goto cleanup;
    if (!writeValue(path, "polarity", "normal")) goto cleanup;
    if (!writeValue(path, "enable", "1")) goto cleanup;

    emit statusMessage("可回收垃圾桶：关盖完成");
    emit commandCompleted();
    return;

cleanup:
    unexportPwmChannel(basePath1, 0);
    emit statusMessage("可回收垃圾桶：关盖操作失败");
    emit commandCompleted();
}

void ServoController::executeRotate2Command()
{
    QMutexLocker locker(&commandMutex);
    emit statusMessage("其它垃圾桶：开盖");

    QString path = basePath2 + "/pwm0";

    if (!exportPwmChannel(basePath2, 0)) {
        emit statusMessage("其它垃圾桶：开盖失败");
        emit commandCompleted();
        return;
    }

    if (!writeValue(path, "period", "20000000")) goto cleanup;
    if (!writeValue(path, "duty_cycle", "500000")) goto cleanup;
    if (!writeValue(path, "polarity", "normal")) goto cleanup;
    if (!writeValue(path, "enable", "1")) goto cleanup;

    emit statusMessage("其它垃圾桶：开盖完成");
    emit commandCompleted();
    return;

cleanup:
    unexportPwmChannel(basePath2, 0);
    emit statusMessage("其它垃圾桶：开盖操作失败");
    emit commandCompleted();
}

void ServoController::executeReset2Command()
{
    QMutexLocker locker(&commandMutex);
    emit statusMessage("其它垃圾桶：关盖");

    QString path = basePath2 + "/pwm0";

    if (!exportPwmChannel(basePath2, 0)) {
        emit statusMessage("其它垃圾桶：关盖失败");
        emit commandCompleted();
        return;
    }

    if (!writeValue(path, "period", "20000000")) goto cleanup;
    if (!writeValue(path, "duty_cycle", "1600000")) goto cleanup;
    if (!writeValue(path, "polarity", "normal")) goto cleanup;
    if (!writeValue(path, "enable", "1")) goto cleanup;

    emit statusMessage("其它垃圾桶：关盖完成");
    emit commandCompleted();
    return;

cleanup:
    unexportPwmChannel(basePath2, 0);
    emit statusMessage("其它垃圾桶：关盖操作失败");
    emit commandCompleted();
}

bool ServoController::isPwmChannelExported(const QString &chipPath, int channel)
{
    QString path = chipPath + "/pwm" + QString::number(channel);
    return QFile::exists(path);
}

bool ServoController::exportPwmChannel(const QString &chipPath, int channel)
{
    if (isPwmChannelExported(chipPath, channel)) {
        return true;
    }
    return writeValue(chipPath, "export", QString::number(channel));
}

bool ServoController::unexportPwmChannel(const QString &chipPath, int channel)
{
    if (!isPwmChannelExported(chipPath, channel)) {
        return true;
    }
    return writeValue(chipPath, "unexport", QString::number(channel));
}

bool ServoController::writeValue(const QString &chipPath, const QString &fileName, const QString &value)
{
    QString fullPath = chipPath + "/" + fileName;
    QFile file(fullPath);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        emit statusMessage("权限错误：" + fullPath);
        return false;
    }

    file.write(value.toUtf8());
    file.close();
    QThread::msleep(10);
    return true;
}
