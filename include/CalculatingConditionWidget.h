#ifndef CALCULATINGCONDITIONWIDGET_H
#define CALCULATINGCONDITIONWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include "MapViewer2D.h"

class CalculatingConditionWidget : public QWidget {
    Q_OBJECT

public:
    explicit CalculatingConditionWidget(QWidget *parent = nullptr);
    ~CalculatingConditionWidget();
    
    BoundaryData getBoundaryData() const;

signals:
    void conditionCommitted(const BoundaryData &boundary);

private slots:
    void onCommitClicked();
    void validateInputs();

private:
    void setupUI();
    
    QLineEdit *xMinEdit;
    QLineEdit *xMaxEdit;
    QLineEdit *yMinEdit;
    QLineEdit *yMaxEdit;
    QLineEdit *depthMinEdit;
    QLineEdit *depthMaxEdit;
    
    QLineEdit *gridSpacingEdit;
    
    QPushButton *commitButton;
};

#endif // CALCULATINGCONDITIONWIDGET_H
