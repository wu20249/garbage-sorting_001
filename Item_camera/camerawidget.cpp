#include "camerawidget.h"
#include <QVBoxLayout>
#include <QMessageBox>
#include <QDebug>
#include <QCameraInfo>

CameraWidget::CameraWidget(const QString &cameraDevice, QWidget *parent)
    : QWidget(parent)
{
    // 通过QCameraInfo查找并设置摄像头设备
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    QCameraInfo selectedCamera = QCameraInfo::defaultCamera();

    // 查找指定的摄像头设备
    bool deviceFound = false;
    for (const QCameraInfo &info : cameras) {
        if (info.deviceName() == cameraDevice) {
            selectedCamera = info;
            deviceFound = true;
            break;
        }
    }

    // 若未找到指定设备，则使用默认摄像头并记录警告
    if (!deviceFound && !cameraDevice.isEmpty()) {
        qWarning() << "指定的摄像头设备未找到:" << cameraDevice
                   << "- 将使用默认摄像头";
    }

    // 使用找到的摄像头信息创建QCamera对象
    camera = new QCamera(selectedCamera, this);

    // 创建取景器
    viewfinder = new QCameraViewfinder(this);

    // 创建图像捕获对象
    imageCapture = new QCameraImageCapture(camera, this);

    // 设置取景器格式
    QCameraViewfinderSettings settings;
    settings.setResolution(640, 480);
    settings.setPixelFormat(QVideoFrame::Format_NV12);
    camera->setViewfinderSettings(settings);

    // 设置取景器
    camera->setViewfinder(viewfinder);

    // 设置布局
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(viewfinder);
    layout->setContentsMargins(0, 0, 0, 0);

    // 连接信号和槽
    connect(imageCapture, &QCameraImageCapture::imageCaptured, this, &CameraWidget::onImageCaptured);
    connect(camera, &QCamera::errorOccurred, this, &CameraWidget::onCameraError);
    connect(imageCapture, QOverload<int, QCameraImageCapture::Error, const QString &>::of(&QCameraImageCapture::error),
            [this](int id, QCameraImageCapture::Error error, const QString &errorString) {
        emit captureFailed(errorString);
        qDebug() << "Image capture error:" << errorString;
    });
}

CameraWidget::~CameraWidget()
{
    if (camera->status() == QCamera::ActiveStatus) {
        camera->stop();
    }
}

bool CameraWidget::startCamera()
{
    camera->start();
    return camera->status() == QCamera::ActiveStatus;
}

void CameraWidget::stopCamera()
{
    camera->stop();
}

bool CameraWidget::isCameraActive() const
{
    return camera->status() == QCamera::ActiveStatus;
}

void CameraWidget::captureImage()
{
    if (isCameraActive()) {
        imageCapture->capture();
    }
}

void CameraWidget::onImageCaptured(int id, const QImage &preview)
{
    Q_UNUSED(id);
    lastCapturedImage = preview;

    // 保存图像到文件
    QString filePath = "/home/elf/captured_image.jpg";
    if (preview.save(filePath, "JPG")) {
        emit imageSaved(filePath);
    } else {
        emit captureFailed("保存图像失败");
    }
}

QImage CameraWidget::getLastCapturedImage() const
{
    return lastCapturedImage;
}

void CameraWidget::onCameraError(QCamera::Error error)
{
    emit captureFailed(camera->errorString());
    qDebug() << "Camera error:" << camera->errorString();
}
