#ifndef VELOCITYMODELWIDGET_H
#define VELOCITYMODELWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QStackedWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>

class VelocityModelWidget : public QWidget {
    Q_OBJECT

public:
    explicit VelocityModelWidget(QWidget *parent = nullptr);
    ~VelocityModelWidget();
    
    QString getModelType() const;

private slots:
    void onModelTypeChanged(int index);
    void onLoad1DModel();
    void onLoad3DModel();

private:
    void setupUI();
    
    QComboBox *modelTypeCombo;
    QStackedWidget *modelStack;
    
    // Homogeneous model
    QWidget *homogeneousWidget;
    QLineEdit *vpHomogeneous;
    
    // 1D model
    QWidget *model1DWidget;
    QPushButton *load1DButton;
    QTextEdit *model1DPreview;
    QString model1DFilePath;
    
    // 3D model
    QWidget *model3DWidget;
    QPushButton *load3DButton;
    QTextEdit *model3DPreview;
    QString model3DFilePath;
};

#endif // VELOCITYMODELWIDGET_H
