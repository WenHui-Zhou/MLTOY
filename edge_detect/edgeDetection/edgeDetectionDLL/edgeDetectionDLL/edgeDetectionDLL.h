// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� EDGEDETECTIONDLL_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// EDGEDETECTIONDLL_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef EDGEDETECTIONDLL_EXPORTS
#define EDGEDETECTIONDLL_API __declspec(dllexport)
#else
#define EDGEDETECTIONDLL_API __declspec(dllimport)
#endif

// �����Ǵ� edgeDetectionDLL.dll ������

extern "C" EDGEDETECTIONDLL_API int detectMain;
