#ifndef METHODWIDGET_H
#define METHODWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QStackedWidget>
#include <QLineEdit>
#include <QScrollArea>

class MethodWidget : public QWidget {
    Q_OBJECT

public:
    explicit MethodWidget(QWidget *parent = nullptr);
    ~MethodWidget();
    
    QString getApproach() const;
    QString getMethod() const;

private slots:
    void onApproachChanged(int index);
    void onLocalMethodChanged(int index);
    void onGlobalMethodChanged(int index);

private:
    void setupUI();
    void createLocalMethodWidgets();
    void createGlobalMethodWidgets();
    
    QComboBox *approachCombo;
    QStackedWidget *methodStack;
    
    // Local approach
    QWidget *localWidget;
    QComboBox *localMethodCombo;
    QStackedWidget *localParamStack;
    
    // Local method parameters
    QWidget *gaussNewtonParams;
    QWidget *steepestDescentParams;
    QWidget *levenbergMarquadtParams;
    
    // Global approach
    QWidget *globalWidget;
    QComboBox *globalMethodCombo;
    QStackedWidget *globalParamStack;
    
    // Global method parameters
    QWidget *gridSearchParams;
    QWidget *simulatedAnnealingParams;
    QWidget *geneticAlgorithmParams;
};

#endif // METHODWIDGET_H
