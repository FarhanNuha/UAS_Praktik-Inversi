#ifndef RESULTWIDGET_H
#define RESULTWIDGET_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>

class ResultWidget : public QWidget {
    Q_OBJECT

public:
    explicit ResultWidget(QWidget *parent = nullptr);
    ~ResultWidget();
    
    void appendResult(const QString &text);
    void clearResults();

private:
    void setupUI();
    
    QTextEdit *resultText;
    QPushButton *clearButton;
    QPushButton *saveButton;
};

#endif // RESULTWIDGET_H
