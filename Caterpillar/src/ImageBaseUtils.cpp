#include "imageBaseUtils.h"


const std::string getCurrentSystemTime()
{
	  auto tt = std::chrono::system_clock::to_time_t
	  (std::chrono::system_clock::now());
	  struct tm* ptm = localtime(&tt);
	  char date[60] = {0};
	  sprintf(date, "%d-%02d-%02d      %02d:%02d:%02d",
	    (int)ptm->tm_year + 1900,(int)ptm->tm_mon + 1,(int)ptm->tm_mday,
	    (int)ptm->tm_hour,(int)ptm->tm_min,(int)ptm->tm_sec);
	  return std::string(date);
}


long LastWishesCrashHandler(EXCEPTION_POINTERS *pException)
{
	String strFileName = getCurrentSystemTime()+".dmp";
	HANDLE hDumpFile = CreateFile((LPCWSTR)strFileName.utf16(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hDumpFile != INVALID_HANDLE_VALUE)
	{
		MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
		dumpInfo.ExceptionPointers = pException;
		dumpInfo.ThreadId = GetCurrentThreadId();
		dumpInfo.ClientPointers = TRUE;
		MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hDumpFile, MiniDumpNormal, &dumpInfo, NULL, NULL);
	}

	EXCEPTION_RECORD* record = pException->ExceptionRecord;
	//QString errCode(QString::number(record->ExceptionCode, 16)), errAdr(QString::number((uint)record->ExceptionAddress, 16)), errMod;
	//QMessageBox::critical(NULL, "error", strFileName, QMessageBox::Ok);
	
	// 使用方法：在主函数里面添加词句就可以在程序崩溃是产生.dmp 的临终遗言
	//SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)ApplicationCrashHandler);
	return EXCEPTION_EXECUTE_HANDLER;
}