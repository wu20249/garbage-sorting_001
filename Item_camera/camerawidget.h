#ifndef CAMERAWIDGET_H
#define CAMERAWIDGET_H

#include <QWidget>
#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraImageCapture>
#include <QImage>

class CameraWidget : public QWidget
{
    Q_OBJECT
public:
    explicit CameraWidget(const QString &cameraDevice = QString(), QWidget *parent = nullptr);
    ~CameraWidget();

    bool startCamera();
    void stopCamera();
    bool isCameraActive() const;
    QImage getLastCapturedImage() const;

signals:
    void imageSaved(const QString &imagePath);
    void captureFailed(const QString &errorMsg);

public slots:
    void captureImage();

private slots:
    void onImageCaptured(int id, const QImage &preview);
    void onCameraError(QCamera::Error error);

private:
    QCamera *camera;
    QCameraViewfinder *viewfinder;
    QCameraImageCapture *imageCapture;
    QImage lastCapturedImage;
};

#endif // CAMERAWIDGET_H
